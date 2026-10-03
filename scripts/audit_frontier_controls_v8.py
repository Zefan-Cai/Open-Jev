"""Independent CPU oracle for new original frontier controls; no model calls."""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from jev.data import validate_records


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, *, nonnegative=False):
    require(type(value) is int and (not nonnegative or value >= 0),
            name + ' must be an exact ' + ('nonnegative ' if nonnegative else '') + 'integer')
    return value


def boolean(value, name):
    require(type(value) is bool, name + ' must be boolean')
    return value


def instant_seconds(value):
    """Parse explicit whole-second instants independently with datetime."""
    require(type(value) is str and re.fullmatch(
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})', value),
        'Explicit whole-second timestamp with UTC offset required')
    require(int(value[11:13]) < 24 and int(value[14:16]) < 60 and int(value[17:19]) < 60,
            'Invalid whole-second clock')
    zone = value[19:]
    require(zone == 'Z' or (int(zone[1:3]) < 24 and int(zone[4:6]) < 60),
            'Invalid UTC offset')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError('Invalid calendar timestamp or UTC offset') from None
    return int((parsed - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds())


def cents_label(cents, currency):
    integer(cents, 'balance')
    require(type(currency) is str and re.fullmatch(r'[A-Z]{3}', currency),
            'Currency must be an explicit three-letter code')
    absolute = abs(cents)
    return f"{currency} {'-' if cents < 0 else ''}{absolute//100}.{absolute%100:02d}"


VERSION = 'frontier-controls-v8'
SPLIT_WIDTHS = {'calibration': [3],
 'ood': [6],
 'test': [5],
 'train': [1, 2],
 'validation': [4]}
POLICIES = {'account_status': 'For each required account, use its greatest-sequence status, ignoring other '
                   'accounts and array order. No matching event means unknown for that component. '
                   'Aggregate closed if any component closed; otherwise paused if any paused; '
                   'otherwise unknown if any missing; otherwise active.',
 'department_route': 'Confirmed fraud on any request line means security review before checking '
                     'policy availability. Otherwise each line requires its greatest valid active '
                     'signed department policy, verified from policy_authority with exact '
                     'department and credential_scope. Missing/withdrawn policy or any line amount '
                     'above its limit means manager review. Approve only when every line has '
                     'active valid policy and is at/below its own limit.',
 'exact_numeric': 'For balance, sum all credit amount_cents and subtract all debit amount_cents in '
                  'ledger. For integer_compare, compute those exact signed sums separately for '
                  'left_ledger and right_ledger, then report below/equal/above. All amounts are '
                  'nonnegative integer USD cents; use exact integer arithmetic and ignore ledger '
                  'array order.',
 'joint_capacity': 'Each required role needs its greatest-sequence verified signed event with '
                   'exact request resource/operation/currency scope. Ignore array order, '
                   'unverified signatures, wrong roles and scopes. Any latest valid revoke means '
                   'reject revoked consent, even if another role is missing. Otherwise missing '
                   'roles mean request missing consent. With all latest records grant, execute '
                   'at/below every capacity; above any capacity means request higher capacity.',
 'latest_authority': 'For each required topic, select its greatest valid revision: verified '
                     'signature from policy_authorities[topic], exact request department in '
                     'department and credential_scope.department, and exact topic in topic and '
                     'credential_scope.topic. Resolve all topics before amount/fraud. Missing '
                     'topic or any latest withdrawn policy means verify policy. Otherwise '
                     'confirmed_fraud means fraud review. With all latest topics active and no '
                     'fraud, automatic processing at/below every topic limit; capacity review '
                     'above any. Ignore array order and invalid records.',
 'refund_priority': "Any item's verified recall means recall remediation before all other checks. "
                    'Otherwise a verified purchase document is required; absent document or any '
                    'item having neither confirmed defect nor intact seal means reject request. If '
                    'every item is eligible, sum all item amount_cents: automatic reimbursement '
                    'at/below automatic_limit_cents; review reimbursement above.',
 'temporal_window': 'Start with recorded_delivery_at. Add seconds for each delay and subtract '
                    'seconds for each advance in delivery_corrections with verified_signature '
                    'true, issuer_role exactly correction_authority, and case_ref exactly this '
                    'case_ref. Corrections are distinct by sequence; ignore array order and '
                    'invalid records. The resulting absolute instant is delivery. An approved '
                    'exception means review. Otherwise accept from delivery through delivery plus '
                    'return_window_seconds inclusive; reject outside. Interpret explicit fixed UTC '
                    'offsets.'}
CONDITIONS = {'account_status': ('active_first',
                    'active_last',
                    'active_history',
                    'paused_first',
                    'paused_last',
                    'paused_history',
                    'closed_first',
                    'closed_last',
                    'closed_history',
                    'unknown_first',
                    'unknown_last',
                    'unknown_all'),
 'department_route': ('approve_below_near',
                      'approve_below_far',
                      'approve_equal',
                      'approve_mixed_below',
                      'manager_above_near',
                      'manager_above_far',
                      'manager_missing',
                      'manager_withdrawn',
                      'security_below',
                      'security_equal',
                      'security_above',
                      'security_without_policy'),
 'exact_numeric': ('balance_negative_near',
                   'balance_negative_far',
                   'balance_zero_small',
                   'balance_zero_large',
                   'balance_positive_near',
                   'balance_positive_far',
                   'compare_below_positive',
                   'compare_below_negative',
                   'compare_equal_positive',
                   'compare_equal_negative',
                   'compare_above_positive',
                   'compare_above_negative'),
 'joint_capacity': ('execute_below_near_first',
                    'execute_below_near_last',
                    'execute_below_far_first',
                    'execute_below_far_last',
                    'execute_equal_first',
                    'execute_equal_last',
                    'execute_regrant_first',
                    'execute_regrant_last',
                    'scope_first_resource',
                    'scope_first_operation',
                    'scope_first_currency',
                    'scope_last_resource',
                    'scope_last_operation',
                    'scope_last_currency',
                    'missing_absent',
                    'missing_invalid',
                    'revoke_first_latest',
                    'revoke_last_latest',
                    'revoke_first_with_missing',
                    'revoke_last_with_missing',
                    'revoke_first_ignore_invalid',
                    'revoke_last_ignore_invalid',
                    'revoke_first_over_capacity',
                    'revoke_last_over_capacity',
                    'capacity_first_near',
                    'capacity_last_near',
                    'capacity_first_far',
                    'capacity_last_far',
                    'capacity_first_reduced',
                    'capacity_last_reduced',
                    'capacity_first_broad',
                    'capacity_last_broad'),
 'latest_authority': ('automatic_first_near',
                      'automatic_last_near',
                      'automatic_first_far',
                      'automatic_last_far',
                      'automatic_first_equal',
                      'automatic_last_equal',
                      'automatic_first_ignore_invalid',
                      'automatic_last_ignore_invalid',
                      'capacity_first_near',
                      'capacity_last_near',
                      'capacity_first_far',
                      'capacity_last_far',
                      'capacity_first_lowered',
                      'capacity_last_lowered',
                      'capacity_first_broad',
                      'capacity_last_broad',
                      'fraud_first_below',
                      'fraud_last_below',
                      'fraud_first_equal',
                      'fraud_last_equal',
                      'fraud_first_above',
                      'fraud_last_above',
                      'fraud_first_broad',
                      'fraud_last_broad',
                      'verify_first_missing',
                      'verify_last_withdrawn',
                      'verify_first_signature',
                      'verify_last_issuer',
                      'verify_first_department',
                      'verify_last_department',
                      'verify_first_topic',
                      'verify_last_topic'),
 'refund_priority': ('automatic_below_near',
                     'automatic_below_far',
                     'automatic_equal',
                     'recall_first_without_document',
                     'recall_last_over_limit',
                     'recall_with_ineligible',
                     'reject_document',
                     'reject_first_item',
                     'reject_last_item',
                     'review_above_near',
                     'review_above_far',
                     'review_broad'),
 'temporal_window': ('at_start',
                     'inside_near',
                     'inside_far',
                     'at_deadline',
                     'before_near',
                     'before_far',
                     'after_near',
                     'after_far',
                     'exception_before',
                     'exception_inside',
                     'exception_deadline',
                     'exception_after')}
OPTIONS = {'account_status': ('active', 'paused', 'closed', 'unknown'),
 'department_route': ('approve', 'manager review', 'security review'),
 'joint_capacity': ('execute',
                    'request missing consent',
                    'reject revoked consent',
                    'request higher capacity'),
 'latest_authority': ('automatic processing', 'capacity review', 'fraud review', 'verify policy'),
 'refund_priority': ('automatic reimbursement',
                     'recall remediation',
                     'reject request',
                     'review reimbursement'),
 'temporal_window': ('accept', 'reject', 'review')}
CHOICE_QUESTIONS = ('Apply the supplied rule to these recorded facts. Which outcome follows?', 'Use the authoritative rule and facts to select the resulting disposition.')
NOUL_TEMPLATE = "Does the supplied rule establish '{proposed_outcome}' for this case?"


def text(value, name):
    require(type(value) is str and bool(value.strip()), name+' must be nonempty text')
    return value


def names(values, name):
    require(type(values) is list and bool(values), name+' must be a nonempty list')
    for value in values:
        text(value, name)
    require(len(set(values)) == len(values), name+' must contain distinct names')
    return values


def family(state):
    require(type(state) is dict, 'Visible state must be an object')
    candidates = [state.get('policy'), state.get('rule'), state.get('trust_contract')]
    trusted = state.get('trusted_policy', {})
    require(type(trusted) is dict, 'Trusted policy must be an object')
    candidates.extend((trusted.get('rules'), trusted.get('ordered_rule')))
    matches = [name for name, policy in POLICIES.items() if policy in candidates]
    require(len(matches) == 1, 'Exactly one reviewed visible policy contract is required')
    text(state.get('case_ref'), 'case_ref')
    return matches[0]


def ledger(entries):
    require(type(entries) is list and bool(entries), 'Ledger must contain operands')
    refs, credits, debits, total = set(), 0, 0, 0
    for entry in entries:
        ref = text(entry['entry_ref'], 'entry_ref')
        require(ref not in refs, 'Duplicate ledger entry reference')
        refs.add(ref)
        cents = integer(entry['amount_cents'], 'ledger cents', nonnegative=True)
        require(cents > 0, 'Ledger operands must be positive, not neutral padding')
        require(entry['direction'] in ('credit', 'debit'), 'Unknown ledger direction')
        if entry['direction'] == 'credit':
            credits += 1
            total += cents
        else:
            debits += 1
            total -= cents
    require(credits == debits and credits > 0, 'Composition needs equal nonempty credit/debit widths')
    if credits > 1:
        credit_values = {entry['amount_cents'] for entry in entries if entry['direction'] == 'credit'}
        debit_values = {entry['amount_cents'] for entry in entries if entry['direction'] == 'debit'}
        require(not credit_values & debit_values, 'Composed ledgers cannot use exact cancelling operand pairs as padding')
    return total, credits


def joint_latest(state):
    request = state['request']
    scope = {key: text(request[key], 'request '+key) for key in ('resource', 'operation', 'currency')}
    require(scope['currency'] == 'USD', 'Authored joint request currency is USD')
    integer(request['amount_cents'], 'request amount', nonnegative=True)
    roles = names(state['trusted_policy']['required_roles'], 'required roles')
    require(len(roles) >= 2, 'At least two essential required roles')
    require(type(state['signed_events']) is list, 'Signed events must be a list')
    latest, seen = {}, set()
    for event in state['signed_events']:
        role = text(event['issuer_role'], 'issuer role')
        signature = boolean(event['verified_signature'], 'signature')
        sequence = integer(event['sequence'], 'credential sequence', nonnegative=True)
        integer(event['capacity_cents'], 'credential capacity', nonnegative=True)
        require(event['status'] in ('grant', 'revoke'), 'Unknown credential status')
        require(type(event['credential_scope']) is dict, 'Credential scope must be an object')
        if signature and role in roles and event['credential_scope'] == scope:
            require((role, sequence) not in seen, 'Conflicting valid role/sequence')
            seen.add((role, sequence))
            if role not in latest or sequence > latest[role]['sequence']:
                latest[role] = event
    return roles, latest


def topic_latest(state):
    request = state['request']
    department = text(request['department'], 'department')
    integer(request['amount_cents'], 'request amount', nonnegative=True)
    boolean(request['confirmed_fraud'], 'confirmed fraud')
    topics = names(state['required_topics'], 'required topics')
    require(len(topics) >= 2 and set(state['policy_authorities']) == set(topics),
            'Every required topic must have one declared issuer')
    for issuer in state['policy_authorities'].values():
        text(issuer, 'topic issuer')
    require(type(state['signed_policy_registry']) is list, 'Policy registry must be a list')
    latest, seen = {}, set()
    for record in state['signed_policy_registry']:
        topic = text(record['topic'], 'policy topic')
        text(record['department'], 'policy department')
        issuer = text(record['issuer_role'], 'policy issuer')
        signature = boolean(record['verified_signature'], 'policy signature')
        revision = integer(record['revision'], 'policy revision', nonnegative=True)
        integer(record['automatic_limit_cents'], 'automatic limit', nonnegative=True)
        require(record['status'] in ('active', 'withdrawn'), 'Unknown policy status')
        require(type(record['credential_scope']) is dict, 'Topic credential scope must be an object')
        if (signature and topic in topics and issuer == state['policy_authorities'][topic]
                and record['department'] == department
                and record['credential_scope'] == {'department': department, 'topic': topic}):
            require((topic, revision) not in seen, 'Conflicting valid topic/revision')
            seen.add((topic, revision))
            if topic not in latest or revision > latest[topic]['revision']:
                latest[topic] = record
    return topics, latest


def department_latest(state):
    authority = text(state['policy_authority'], 'policy authority')
    require(type(state['signed_policies']) is list, 'Department policies must be a list')
    latest, seen = {}, set()
    for record in state['signed_policies']:
        department = text(record['department'], 'policy department')
        issuer = text(record['issuer_role'], 'policy issuer')
        signature = boolean(record['verified_signature'], 'policy signature')
        revision = integer(record['revision'], 'policy revision', nonnegative=True)
        integer(record['approval_limit_cents'], 'approval limit', nonnegative=True)
        require(record['status'] in ('active', 'withdrawn'), 'Unknown department policy status')
        if signature and issuer == authority and record['credential_scope'] == department:
            require((department, revision) not in seen, 'Conflicting valid department/revision')
            seen.add((department, revision))
            if department not in latest or revision > latest[department]['revision']:
                latest[department] = record
    return latest


def facts(state):
    name = family(state)
    result = {'family': name}
    if name == 'temporal_window':
        anchor = instant_seconds(state['recorded_delivery_at'])
        issuer = text(state['correction_authority'], 'correction authority')
        require(type(state['delivery_corrections']) is list, 'Delivery corrections must be a list')
        applicable, seen = [], set()
        for correction in state['delivery_corrections']:
            signature = boolean(correction['verified_signature'], 'correction signature')
            text(correction['issuer_role'], 'correction issuer')
            text(correction['case_ref'], 'correction case')
            sequence = integer(correction['sequence'], 'correction sequence', nonnegative=True)
            seconds = integer(correction['seconds'], 'correction seconds', nonnegative=True)
            require(seconds > 0 and correction['direction'] in ('delay', 'advance'),
                    'Correction must have a nonzero valid direction')
            if signature and correction['issuer_role'] == issuer and correction['case_ref'] == state['case_ref']:
                require(sequence not in seen, 'Conflicting applicable correction sequence')
                seen.add(sequence)
                applicable.append(correction)
                anchor += seconds if correction['direction'] == 'delay' else -seconds
        require(bool(applicable), 'Temporal composition needs applicable nonzero corrections')
        window = integer(state['return_window_seconds'], 'window', nonnegative=True)
        require(window > 0, 'Return window must be positive')
        exception = boolean(state['exception_approved'], 'approved exception')
        elapsed = instant_seconds(state['request_received_at']) - anchor
        result.update(width=len(applicable), elapsed=elapsed, window=window, exception=exception,
                      outcome='review' if exception else 'accept' if 0 <= elapsed <= window else 'reject')
    elif name == 'exact_numeric':
        require(state['currency'] == 'USD', 'Numeric contract uses USD cents')
        require(state['task'] in ('balance', 'integer_compare'), 'Unknown numeric task')
        if state['task'] == 'balance':
            total, width = ledger(state['ledger'])
            result.update(width=width, balance=total, outcome=cents_label(total, 'USD'))
        else:
            left, width = ledger(state['left_ledger'])
            right, other_width = ledger(state['right_ledger'])
            require(width == other_width, 'Left/right ledger composition widths differ')
            result.update(width=width, left=left, right=right, margin=left-right,
                          outcome='below' if left < right else 'above' if left > right else 'equal')
    elif name == 'joint_capacity':
        roles, latest = joint_latest(state)
        missing = [index for index, role in enumerate(roles) if role not in latest]
        revoked = [index for index, role in enumerate(roles) if role in latest and latest[role]['status'] == 'revoke']
        margin = None if missing or revoked else state['request']['amount_cents'] - min(record['capacity_cents'] for record in latest.values())
        result.update(width=len(roles)-1, missing=missing, revoked=revoked, margin=margin,
                      outcome='reject revoked consent' if revoked else 'request missing consent' if missing
                      else 'request higher capacity' if margin > 0 else 'execute')
    elif name == 'latest_authority':
        topics, latest = topic_latest(state)
        missing = [index for index, topic in enumerate(topics) if topic not in latest]
        withdrawn = [index for index, topic in enumerate(topics) if topic in latest and latest[topic]['status'] == 'withdrawn']
        margin = None if missing or withdrawn else state['request']['amount_cents'] - min(record['automatic_limit_cents'] for record in latest.values())
        fraud = state['request']['confirmed_fraud']
        result.update(width=len(topics)-1, missing=missing, withdrawn=withdrawn, margin=margin, fraud=fraud,
                      outcome='verify policy' if missing or withdrawn else 'fraud review' if fraud
                      else 'capacity review' if margin > 0 else 'automatic processing')
    elif name == 'account_status':
        accounts = names(state['required_accounts'], 'required accounts')
        require(type(state['events']) is list, 'Account events must be a list')
        latest, seen = {}, set()
        for event in state['events']:
            account = text(event['account'], 'event account')
            sequence = integer(event['sequence'], 'account sequence', nonnegative=True)
            require(event['status'] in ('active', 'paused', 'closed'), 'Unknown account status')
            if account in accounts:
                require((account, sequence) not in seen, 'Conflicting valid account/sequence')
                seen.add((account, sequence))
                if account not in latest or sequence > latest[account]['sequence']:
                    latest[account] = event
        statuses = [latest[account]['status'] if account in latest else 'unknown' for account in accounts]
        result.update(width=len(accounts), statuses=statuses,
                      outcome=next(status for status in ('closed', 'paused', 'unknown', 'active') if status in statuses))
    elif name == 'department_route':
        lines = state['request_lines']
        require(type(lines) is list and bool(lines), 'At least one request line')
        for line in lines:
            text(line['department'], 'line department')
            integer(line['amount_cents'], 'line amount', nonnegative=True)
            boolean(line['confirmed_fraud'], 'line fraud')
        latest = department_latest(state)
        blocked = any(line['department'] not in latest or latest[line['department']]['status'] == 'withdrawn' for line in lines)
        margins = [line['amount_cents']-latest[line['department']]['approval_limit_cents'] for line in lines
                   if line['department'] in latest and latest[line['department']]['status'] == 'active']
        fraud = any(line['confirmed_fraud'] for line in lines)
        result.update(width=len(lines), blocked=blocked, margins=margins, fraud=fraud,
                      outcome='security review' if fraud else 'manager review' if blocked or any(value > 0 for value in margins) else 'approve')
    else:
        document = boolean(state['purchase_document_verified'], 'purchase document')
        items = state['items']
        require(type(items) is list and bool(items), 'At least one refund item')
        refs, total, recall, eligible = set(), 0, False, True
        for item in items:
            ref = text(item['item_ref'], 'item reference')
            require(ref not in refs, 'Duplicate refund item reference')
            refs.add(ref)
            cents = integer(item['amount_cents'], 'item amount', nonnegative=True)
            require(cents > 0, 'Refund items cannot have neutral zero amounts')
            total += cents
            recall = boolean(item['verified_recall'], 'verified recall') or recall
            defect = boolean(item['defect_confirmed'], 'confirmed defect')
            sealed = boolean(item['seal_intact'], 'intact seal')
            eligible = eligible and (defect or sealed)
        require(state['trusted_policy']['currency'] == 'USD', 'Refund contract uses USD cents')
        limit = integer(state['trusted_policy']['automatic_limit_cents'], 'refund limit', nonnegative=True)
        result.update(width=len(items), total=total, margin=total-limit, recall=recall, document=document, eligible=eligible,
                      outcome='recall remediation' if recall else 'reject request' if not document or not eligible
                      else 'review reimbursement' if total > limit else 'automatic reimbursement')
    return result


def outcome(state):
    return facts(state)['outcome']


def money_value(label):
    match = re.fullmatch(r'USD (-?)(\d+)\.(\d{2})', label)
    require(match is not None, 'Balance candidate must be an exact USD amount')
    cents = (int(match[2])*100+int(match[3])) * (-1 if match[1] else 1)
    require(cents_label(cents, 'USD') == label, 'Balance candidate must be canonical')
    return cents


def gold(row):
    derived = facts(row['state'])
    if row['kind'] == 'choice':
        require(row['question'] in CHOICE_QUESTIONS, 'Unreviewed Choice question')
        if derived['family'] == 'exact_numeric' and row['state']['task'] == 'balance':
            require(len(row['options']) == 5 and len(set(row['options'])) == 5, 'Five distinct balance candidates')
            for option in row['options']:
                money_value(option)
        else:
            expected = OPTIONS.get(derived['family'], ('below', 'equal', 'above'))
            require(set(row['options']) == set(expected) and len(row['options']) == len(expected), 'Choice options differ from visible contract')
        require(derived['outcome'] in row['options'], 'Actual outcome missing from Choice options')
        return derived['outcome']
    require(row['kind'] == 'noul' and row['options'] == ['no', 'yes'], 'Only reviewed Choice/Noul are supported')
    match = re.fullmatch(r"Does the supplied rule establish '([^']+)' for this case\?", row['question'])
    require(match is not None, 'One reviewed visible Noul proposition required')
    proposal = match[1]
    if derived['family'] == 'exact_numeric' and row['state']['task'] == 'balance':
        money_value(proposal)
    else:
        require(proposal in OPTIONS.get(derived['family'], ('below', 'equal', 'above')), 'Noul proposition outside visible action contract')
    return 'yes' if proposal == derived['outcome'] else 'no'


def check_condition(row, derived):
    """Check annotations against facts after deriving gold independently."""
    state, name, condition = row['state'], derived['family'], row['metadata']['condition']
    valid = False
    if name == 'temporal_window':
        delta, window, exception = derived['elapsed'], derived['window'], derived['exception']
        tests = {'at_start':delta == 0, 'inside_near':delta == 1, 'inside_far':1 < delta < window,
                 'at_deadline':delta == window, 'before_near':delta == -1, 'before_far':delta < -1,
                 'after_near':delta-window == 1, 'after_far':delta-window > 1,
                 'exception_before':delta < 0, 'exception_inside':0 < delta < window,
                 'exception_deadline':delta == window, 'exception_after':delta > window}
        valid = tests.get(condition, False) and exception == condition.startswith('exception_')
    elif name == 'exact_numeric':
        if condition.startswith('balance_'):
            total = derived.get('balance')
            valid = total is not None and (total == 0 if 'zero' in condition else total < 0 if 'negative' in condition else total > 0)
            if valid and condition.endswith('_near'):
                valid = abs(total) == 1
            if valid and condition.endswith('_far'):
                valid = abs(total) > 1
        else:
            relation = condition.split('_')[1]
            valid = state['task'] == 'integer_compare' and derived['outcome'] == relation
            valid = valid and (derived['left'] < 0 and derived['right'] < 0 if condition.endswith('_negative') else derived['left'] > 0 and derived['right'] > 0)
    elif name == 'joint_capacity':
        roles, latest = joint_latest(state)
        position = 0 if '_first' in condition else len(roles)-1 if '_last' in condition else None
        role = roles[position] if position is not None else None
        margin = derived['margin']
        if condition.startswith('execute_'):
            valid = derived['outcome'] == 'execute'
            if '_near_' in condition:valid = valid and margin == -1
            elif '_far_' in condition:valid = valid and margin < -1
            elif '_equal_' in condition:valid = valid and margin == 0
            else:
                valid = valid and any(event['issuer_role'] == role and event['status'] == 'revoke'
                    and event['verified_signature'] is True and event['credential_scope'] == latest[role]['credential_scope']
                    and event['sequence'] < latest[role]['sequence'] for event in state['signed_events'])
        elif condition.startswith('scope_'):
            field = condition.rsplit('_', 1)[1]
            scope = {key: state['request'][key] for key in ('resource', 'operation', 'currency')}
            valid = derived['outcome'] == 'request missing consent' and derived['missing'] == [position]
            valid = valid and any(event['issuer_role'] == role and event['verified_signature'] is True
                and [key for key in scope if event['credential_scope'].get(key) != scope[key]] == [field]
                for event in state['signed_events'])
        elif condition.startswith('missing_'):
            valid = derived['outcome'] == 'request missing consent' and bool(derived['missing'])
            if condition == 'missing_absent':
                valid = valid and any(not any(event['issuer_role'] == roles[index] for event in state['signed_events']) for index in derived['missing'])
            else:
                valid = valid and bool(state['signed_events'])
        elif condition.startswith('revoke_'):
            valid = derived['outcome'] == 'reject revoked consent' and position in derived['revoked']
            if condition.endswith('_with_missing'):valid = valid and bool(derived['missing'])
            elif condition.endswith('_ignore_invalid'):
                valid = valid and any(event['issuer_role'] == role and event['status'] == 'grant'
                    and event['sequence'] > latest[role]['sequence'] and (event['verified_signature'] is False
                    or event['credential_scope'] != latest[role]['credential_scope']) for event in state['signed_events'])
            elif condition.endswith('_over_capacity'):
                valid = valid and state['request']['amount_cents'] > min(event['capacity_cents'] for event in latest.values())
        else:
            valid = derived['outcome'] == 'request higher capacity' and latest[role]['capacity_cents'] < state['request']['amount_cents']
            if condition.endswith('_near'):valid = valid and margin == 1
            elif condition.endswith(('_far', '_broad')):valid = valid and margin > 1
            else:
                valid = valid and any(event['issuer_role'] == role and event['status'] == 'grant'
                    and event['sequence'] < latest[role]['sequence'] and event['capacity_cents'] >= state['request']['amount_cents']
                    and event['verified_signature'] is True and event['credential_scope'] == latest[role]['credential_scope']
                    for event in state['signed_events'])
    elif name == 'latest_authority':
        topics, latest = topic_latest(state)
        position = 0 if '_first' in condition else len(topics)-1
        topic = topics[position]
        margin = derived['margin']
        if condition.startswith('automatic_'):
            valid = derived['outcome'] == 'automatic processing'
            if condition.endswith('_near'):valid = valid and margin == -1
            elif condition.endswith('_far'):valid = valid and margin < -1
            elif condition.endswith('_equal'):valid = valid and margin == 0
            else:
                valid = valid and any(record['topic'] == topic and record['revision'] > latest[topic]['revision']
                    and (record['verified_signature'] is False or record['issuer_role'] != state['policy_authorities'][topic]
                    or record['credential_scope'] != latest[topic]['credential_scope'] or record['department'] != state['request']['department'])
                    for record in state['signed_policy_registry'])
        elif condition.startswith('capacity_'):
            valid = derived['outcome'] == 'capacity review' and latest[topic]['automatic_limit_cents'] < state['request']['amount_cents']
            if condition.endswith('_near'):valid = valid and margin == 1
            elif condition.endswith(('_far', '_broad')):valid = valid and margin > 1
            else:
                valid = valid and any(record['topic'] == topic and record['revision'] < latest[topic]['revision']
                    and record['verified_signature'] is True and record['issuer_role'] == state['policy_authorities'][topic]
                    and record['credential_scope'] == latest[topic]['credential_scope']
                    and record['automatic_limit_cents'] >= state['request']['amount_cents'] for record in state['signed_policy_registry'])
        elif condition.startswith('fraud_'):
            valid = derived['outcome'] == 'fraud review'
            valid = valid and (margin < 0 if condition.endswith('_below') else margin == 0 if condition.endswith('_equal') else margin > 0)
        else:
            valid = derived['outcome'] == 'verify policy' and position in derived['missing']+derived['withdrawn']
            if condition.endswith('_withdrawn'):valid = valid and position in derived['withdrawn']
            elif condition.endswith('_missing'):
                valid = valid and not any(record['topic'] == topic for record in state['signed_policy_registry'])
            else:
                candidates = [record for record in state['signed_policy_registry'] if record['topic'] == topic or record['issuer_role'] == state['policy_authorities'][topic]]
                if condition.endswith('_signature'):valid = valid and any(record['verified_signature'] is False for record in candidates)
                elif condition.endswith('_issuer'):valid = valid and any(record['issuer_role'] != state['policy_authorities'][topic] for record in candidates)
                elif condition.endswith('_department'):valid = valid and any(record['department'] != state['request']['department'] or record['credential_scope'].get('department') != state['request']['department'] for record in candidates)
                else:valid = valid and any(record['topic'] != topic or record['credential_scope'].get('topic') != topic for record in candidates)
    elif name == 'account_status':
        disposition = condition.split('_')[0]
        valid = derived['outcome'] == disposition
        if condition.endswith('_all'):valid = valid and all(status == 'unknown' for status in derived['statuses'])
        elif condition.endswith('_first'):valid = valid and derived['statuses'][0] == disposition
        elif condition.endswith('_last'):valid = valid and derived['statuses'][-1] == disposition
        else:
            counts = Counter(event['account'] for event in state['events'] if event['account'] in state['required_accounts'])
            valid = valid and any(count > 1 for count in counts.values())
    elif name == 'department_route':
        disposition = condition.split('_')[0]
        valid = derived['outcome'] == {'approve':'approve', 'manager':'manager review', 'security':'security review'}[disposition]
        if condition.endswith('_missing'):valid = valid and any(line['department'] not in department_latest(state) for line in state['request_lines'])
        elif condition.endswith('_withdrawn'):valid = valid and any(record['status'] == 'withdrawn' for record in department_latest(state).values())
        elif condition.endswith('_without_policy'):valid = valid and derived['blocked']
        elif condition.endswith('_equal'):valid = valid and max(derived['margins']) == 0
        elif condition.endswith('_near'):valid = valid and max(derived['margins']) == (-1 if disposition == 'approve' else 1)
        elif condition.endswith('_far'):valid = valid and (max(derived['margins']) < -1 if disposition == 'approve' else max(derived['margins']) > 1)
        elif condition.endswith('_below'):valid = valid and max(derived['margins']) < 0
        elif condition.endswith('_above'):valid = valid and max(derived['margins']) > 0
    else:
        disposition = condition.split('_')[0]
        valid = derived['outcome'] == {'automatic':'automatic reimbursement', 'recall':'recall remediation', 'reject':'reject request', 'review':'review reimbursement'}[disposition]
        # First/last annotations refer to the canonical forward fact case.
        # audit_rows independently checks the claimed layout reversal below.
        first_index = 0 if row['metadata']['layout'] == 'forward' else -1
        last_index = -1 if first_index == 0 else 0
        if condition.endswith('_near'):valid = valid and derived['margin'] == (-1 if disposition == 'automatic' else 1)
        elif condition.endswith(('_far', '_broad')):valid = valid and (derived['margin'] < -1 if disposition == 'automatic' else derived['margin'] > 1)
        elif condition.endswith('_equal'):valid = valid and derived['margin'] == 0
        elif condition == 'recall_first_without_document':valid = valid and state['items'][first_index]['verified_recall'] and not derived['document']
        elif condition == 'recall_last_over_limit':valid = valid and state['items'][last_index]['verified_recall'] and derived['margin'] > 0
        elif condition == 'recall_with_ineligible':valid = valid and not derived['eligible']
        elif condition == 'reject_document':valid = valid and not derived['document']
        else:
            item = state['items'][first_index if condition == 'reject_first_item' else last_index]
            valid = valid and not item['defect_confirmed'] and not item['seal_intact']
    require(valid, 'Condition annotation disagrees with independent facts: '+row['id']+'/'+condition)


def essentiality(group, name):
    """Check each composition input can change a result in an authored case."""
    conditions = {row['metadata']['condition']:row for row in group if row['kind'] == 'choice'}
    tested = 0
    if name == 'temporal_window':
        baseline = conditions['at_start']['state']
        for correction in baseline['delivery_corrections']:
            if not (correction['verified_signature'] is True and correction['issuer_role'] == baseline['correction_authority'] and correction['case_ref'] == baseline['case_ref']):continue
            state = copy.deepcopy(conditions['at_start' if correction['direction'] == 'delay' else 'at_deadline']['state'])
            selected = next(record for record in state['delivery_corrections']
                            if record['sequence'] == correction['sequence'] and record['verified_signature'] is True
                            and record['issuer_role'] == state['correction_authority'] and record['case_ref'] == state['case_ref'])
            before = outcome(state)
            selected['seconds'] += 1
            require(outcome(state) != before, 'Applicable correction is not outcome-essential')
            tested += 1
    elif name == 'exact_numeric':
        for condition, fields in (('balance_zero_small', ('ledger',)), ('compare_equal_positive', ('left_ledger', 'right_ledger'))):
            baseline = conditions[condition]['state']
            for field in fields:
                for index in range(len(baseline[field])):
                    state = copy.deepcopy(baseline)
                    selected = state[field][index]
                    opposite = {record['amount_cents'] for record in state[field]
                                if record['direction'] != selected['direction']}
                    changed = selected['amount_cents']+1
                    while changed in opposite:changed += 1
                    selected['amount_cents'] = changed
                    require(outcome(state) != outcome(baseline), 'Ledger operand is not outcome-essential')
                    tested += 1
    elif name == 'joint_capacity':
        baseline = conditions['execute_equal_first']['state']
        roles, current = joint_latest(baseline)
        for role in roles:
            state = copy.deepcopy(baseline)
            next(record for record in state['signed_events'] if record == current[role])['status'] = 'revoke'
            require(outcome(state) != outcome(baseline), 'Required role is not outcome-essential')
            tested += 1
    elif name == 'latest_authority':
        baseline = conditions['automatic_first_equal']['state']
        topics, current = topic_latest(baseline)
        for topic in topics:
            state = copy.deepcopy(baseline)
            next(record for record in state['signed_policy_registry'] if record == current[topic])['status'] = 'withdrawn'
            require(outcome(state) != outcome(baseline), 'Required topic is not outcome-essential')
            tested += 1
    elif name == 'account_status':
        baseline = conditions['active_first']['state']
        for account in baseline['required_accounts']:
            state = copy.deepcopy(baseline)
            max((record for record in state['events'] if record['account'] == account),key=lambda record:record['sequence'])['status'] = 'closed'
            require(outcome(state) != outcome(baseline), 'Required account is not outcome-essential')
            tested += 1
    elif name == 'department_route':
        baseline = conditions['approve_equal']['state']
        current = department_latest(baseline)
        for index, line in enumerate(baseline['request_lines']):
            state = copy.deepcopy(baseline)
            state['request_lines'][index]['amount_cents'] = current[line['department']]['approval_limit_cents']+1
            require(outcome(state) != outcome(baseline), 'Request line is not outcome-essential')
            tested += 1
    else:
        baseline = conditions['automatic_equal']['state']
        for index in range(len(baseline['items'])):
            state = copy.deepcopy(baseline)
            state['items'][index]['verified_recall'] = True
            require(outcome(state) != outcome(baseline), 'Refund item is not outcome-essential')
            tested += 1
    return tested


def cells(counter, keys):
    return [{**dict(zip(keys, key)), 'rows':value} for key,value in sorted(counter.items())]


def reversed_record_state(state):
    transformed = copy.deepcopy(state)
    for field, values in transformed.items():
        if type(values) is list and field not in ('required_roles','required_topics','required_accounts'):
            values.reverse()
    return transformed


def same_typed_state(expected, actual):
    """Match reviewed-schema presentations with one bijective identity map.

    Record order must already be the permitted declared layout. Required-name
    order anchors the map; literal/typed facts and every reference equality
    remain fixed. This bounded matcher performs no graph-isomorphism search.
    """
    identifiers = {'case_ref','correction_authority','issuer_role','resource','operation',
                   'department','policy_authority','topic','account','item_ref','entry_ref',
                   'credential_scope','required_roles','required_topics','required_accounts'}
    forward, inverse = {}, {}
    def identity(left, right):
        if type(left) is not str or type(right) is not str:return False
        if left in forward:return forward[left] == right
        if right in inverse:return False
        forward[left], inverse[right] = right,left
        return True
    def compare(left, right, field=''):
        if type(left) is not type(right):return False
        if isinstance(left,dict):
            if field == 'policy_authorities':
                if not all(key in forward for key in left):return False
                if {forward[key] for key in left} != set(right):return False
                return all(identity(value,right[forward[key]]) for key,value in left.items())
            return set(left) == set(right) and all(compare(value,right[key],key) for key,value in left.items())
        if isinstance(left,list):
            return len(left) == len(right) and all(compare(a,b,field) for a,b in zip(left,right))
        if type(left) is str:
            if field in identifiers:return identity(left,right)
            if field in ('recorded_delivery_at','request_received_at'):
                return instant_seconds(left) == instant_seconds(right)
        return left == right
    if type(expected) is not dict or type(actual) is not dict:return False
    for field in ('required_topics','required_accounts'):
        if field in expected or field in actual:
            if field not in expected or field not in actual or not compare(expected[field],actual[field],field):return False
    expected_roles = expected.get('trusted_policy',{}).get('required_roles')
    actual_roles = actual.get('trusted_policy',{}).get('required_roles')
    if expected_roles is not None or actual_roles is not None:
        if not compare(expected_roles,actual_roles,'required_roles'):return False
    return compare(expected,actual)


def semantic_state(state, *, sort_records=True):
    """Normalize declared identities/order; retain other names injectively.

    This is conservative for unrecognized identity renaming. It never merges
    foreign references merely because the policy ignores them.
    """
    name = family(state)
    substitutions = {state['case_ref']:'this_case'}
    if name == 'temporal_window':
        substitutions[state['correction_authority']] = 'correction_authority'
    elif name == 'joint_capacity':
        substitutions.update({role:'required_role_'+str(index) for index,role in enumerate(state['trusted_policy']['required_roles'])})
        substitutions.update({state['request'][key]:'request_'+key for key in ('resource','operation')})
    elif name == 'latest_authority':
        substitutions[state['request']['department']] = 'request_department'
        substitutions.update({topic:'required_topic_'+str(index) for index,topic in enumerate(state['required_topics'])})
        for index, topic in enumerate(state['required_topics']):
            issuer = state['policy_authorities'][topic]
            substitutions.setdefault(issuer,'topic_authority_'+str(index))
    elif name == 'account_status':
        substitutions.update({account:'required_account_'+str(index) for index,account in enumerate(state['required_accounts'])})
    elif name == 'department_route':
        substitutions[state['policy_authority']] = 'policy_authority'
        departments = {line['department'] for line in state['request_lines']}
        def component(department):
            lines = sorted((line['amount_cents'],line['confirmed_fraud']) for line in state['request_lines'] if line['department'] == department)
            policies = sorted((record['revision'],record['status'],record['approval_limit_cents'],
                record['verified_signature'],record['issuer_role'] == state['policy_authority'],
                record['credential_scope'] == department) for record in state['signed_policies'] if record['department'] == department)
            return lines,policies
        ordered = sorted(departments,key=lambda department:(component(department),department))
        substitutions.update({department:'department_'+str(index) for index,department in enumerate(ordered)})
    def normalize(value, key=''):
        if isinstance(value, dict):
            return {substitutions.get(field, field):normalize(item, field) for field,item in value.items()
                    if field not in ('entry_ref','item_ref')}
        if isinstance(value, list):
            items = [normalize(item, key) for item in value]
            return items if not sort_records or key in ('required_roles','required_topics','required_accounts') else sorted(items,key=lambda item:json.dumps(item,sort_keys=True))
        if type(value) is str:
            if key in ('recorded_delivery_at','request_received_at'):
                return instant_seconds(value)
            if value in substitutions:return substitutions[value]
            return value
        return value
    return json.dumps(normalize(state),sort_keys=True,separators=(',',':'),allow_nan=False)


def audit_rows(records):
    schema = validate_records(records)
    groups = defaultdict(list)
    decisions, actions, conditions, layouts, positions, widths, numeric = (Counter() for _ in range(7))
    scope_cross, topic_cross, money_ranks = Counter(), Counter(), Counter()
    blocked = Counter()
    for row in records:
        require(row['source'] == VERSION and row['metadata']['provenance']['generator_version'] == VERSION,
                'Only new original v8 rows may enter this audit')
        provenance = row['metadata']['provenance']
        require(provenance['license'] == 'CC0-1.0' and provenance['type'] == 'synthetic'
                and type(provenance['seed']) is int and provenance['seed'] == 20261003,
                'Expected declared original CC0 synthetic provenance and seed')
        derived = facts(row['state'])
        name, width, split = derived['family'], derived['width'], row['split']
        require(row['metadata']['scenario_family'] == name, 'Metadata family disagrees with visible policy')
        declared_width = integer(row['metadata']['composition_width'],'declared composition width',nonnegative=True)
        require(width in SPLIT_WIDTHS[split] and declared_width == width,
                'Essential visible composition width differs from split declaration')
        if name == 'temporal_window':
            applicable = sorted((correction for correction in row['state']['delivery_corrections']
                if correction['verified_signature'] is True and correction['issuer_role'] == row['state']['correction_authority']
                and correction['case_ref'] == row['state']['case_ref']),key=lambda correction:correction['sequence'])
            require(all(left['direction'] != right['direction'] for left,right in zip(applicable,applicable[1:])),
                    'Declared temporal scaffold requires alternating applicable corrections')
        decision = gold(row)
        require(row['target'] == [float(option == decision) for option in row['options']],
                'Independent state-derived target differs: '+row['id'])
        condition = row['metadata']['condition']
        require(condition in CONDITIONS[name], 'Unknown authored condition')
        check_condition(row, derived)
        case_index, presentation = row['metadata']['case_index'], row['metadata']['presentation_index']
        integer(case_index, 'case index', nonnegative=True)
        integer(presentation, 'presentation index', nonnegative=True)
        require(case_index < len(CONDITIONS[name]) and CONDITIONS[name][case_index] == condition
                and presentation in range(4) and row['metadata']['provenance']['variant'] == case_index*4+presentation,
                'Case/presentation annotations disagree')
        layout = row['metadata']['layout']
        require(layout in ('forward','reverse'), 'Unknown record layout')
        if presentation < 2:
            require(row['kind'] == 'choice' and row['question'] == CHOICE_QUESTIONS[presentation]
                    and layout == ('forward' if presentation == 0 else 'reverse'), 'Choice presentation must cross layouts')
        else:
            require(row['kind'] == 'noul' and decision == ('yes' if presentation == 2 else 'no'),
                    'Every case needs both visible Noul truths')
        kind_truth = row['kind'] if row['kind'] == 'choice' else 'noul/'+decision
        if name == 'exact_numeric' and row['state']['task'] == 'balance' and row['kind'] == 'choice':
            actual_rank = sorted(money_value(option) for option in row['options']).index(derived['balance'])
            require(integer(row['metadata']['money_gold_numeric_rank'],'money numeric rank',nonnegative=True) == actual_rank,
                    'Money-rank annotation disagrees with numeric candidate values')
            if presentation == 0:money_ranks[split,width,actual_rank] += 1
        groups[row['group_id']].append(row)
        decisions[split,name,row['kind'],decision] += 1
        actions[split,name,row['kind'],derived['outcome']] += 1
        conditions[split,name,condition,kind_truth] += 1
        layouts[split,name,condition,kind_truth,layout] += 1
        positions[split,name,row['kind'],len(row['options']),row['options'].index(decision)] += 1
        widths[split,name,width] += 1
        for quantity in ('margin','balance','elapsed'):
            if quantity in derived and derived[quantity] is not None:
                numeric[split,name,quantity,derived[quantity]] += 1
        for value in derived.get('margins', []):
            numeric[split,name,'active_line_margin',value] += 1
        if name == 'temporal_window' and not 0 <= derived['elapsed'] <= derived['window']:
            value = derived['elapsed'] if derived['elapsed'] < 0 else derived['elapsed']-derived['window']
            numeric[split,name,'outside_margin_seconds',value] += 1
        if name in ('joint_capacity','latest_authority') and derived['margin'] is None:
            blocked[split,name,derived['outcome']] += 1
        if name == 'joint_capacity' and condition.startswith('scope_'):
            role = 0 if '_first_' in condition else width
            require(integer(row['metadata']['affected_role_index'],'affected role',nonnegative=True) == role
                    and row['metadata']['scope_field'] == condition.rsplit('_',1)[1],
                    'Scope-axis annotation differs from missing-role facts')
            scope_cross[split,condition.rsplit('_',1)[1],'first' if role == 0 else 'last',kind_truth] += 1
        if name == 'latest_authority' and condition.startswith('verify_') and condition.endswith(('_department','_topic')):
            topic = 0 if '_first_' in condition else width
            require(integer(row['metadata']['affected_topic_index'],'affected topic',nonnegative=True) == topic,
                    'Topic-axis annotation differs from facts')
            topic_cross[split,condition.rsplit('_',1)[1],'first' if topic == 0 else 'last',kind_truth] += 1
    require(schema['records'] == 2976 and schema['groups'] == 42,
            'Full declaration requires 2976 rows and 42 parent groups')
    require(schema['splits'] == {'train':992,'calibration':496,'validation':496,'test':496,'ood':496},
            'All five complete split denominators required')
    parent_keys, essential_checks, equivalents, noul_layout_shortfalls, order_collapses = set(), 0, [], [], []
    integrity_comparisons = 0
    for identity, group in groups.items():
        first = group[0]
        name, width, split = family(first['state']), facts(first['state'])['width'], first['split']
        key = split,name,width
        require(key not in parent_keys, 'One parent scaffold per family/width required')
        parent_keys.add(key)
        require(all(row['split'] == split and family(row['state']) == name and facts(row['state'])['width'] == width for row in group), 'Parent group mixes split/family/width')
        require(len(group) == 4*len(CONDITIONS[name]) and {row['metadata']['provenance']['variant'] for row in group} == set(range(4*len(CONDITIONS[name]))), 'Incomplete or duplicate parent presentations')
        cases = defaultdict(list)
        for row in group:cases[row['metadata']['case_index']].append(row)
        signatures = defaultdict(list)
        for case, case_rows in cases.items():
            require(len(case_rows) == 4 and {row['metadata']['presentation_index'] for row in case_rows} == set(range(4)), 'Incomplete fact-case presentations')
            presentations = {row['metadata']['presentation_index']:row for row in case_rows}
            if name == 'exact_numeric' and presentations[0]['state']['task'] == 'balance':
                candidate_values = sorted(money_value(option) for option in presentations[0]['options'])
                require(candidate_values == sorted(money_value(option) for option in presentations[1]['options']),
                        'Choice presentations must preserve the same amount candidate set')
                rank = candidate_values.index(facts(presentations[0]['state'])['balance'])
                require(all(integer(row['metadata']['money_gold_numeric_rank'],'money numeric rank',nonnegative=True) == rank for row in case_rows),
                        'Every amount presentation must retain the derived numeric rank')
            forward = presentations[0]['state']
            reverse = reversed_record_state(forward)
            require(same_typed_state(reverse,presentations[1]['state']),
                    'Claimed reverse Choice layout does not reverse actual records')
            integrity_comparisons += 1
            for row in [item for item in case_rows if item['kind'] == 'noul']:
                expected_state = forward if row['metadata']['layout'] == 'forward' else reverse
                require(same_typed_state(expected_state,row['state']),
                        'Claimed Noul layout changed actual records/references')
                integrity_comparisons += 1
            if semantic_state(forward,sort_records=False) == semantic_state(reverse,sort_records=False):
                order_collapses.append({'split':split,'family':name,'width':width,'condition':CONDITIONS[name][case]})
            noul = [row for row in case_rows if row['kind'] == 'noul']
            require(len({row['metadata']['layout'] for row in noul}) == 1, 'Noul truth must not select layout')
            absent = ({'forward','reverse'}-{row['metadata']['layout'] for row in noul}).pop()
            noul_layout_shortfalls.append({'split':split,'family':name,'width':width,'condition':CONDITIONS[name][case], 'unobserved_noul_layout':absent})
            base = min(case_rows,key=lambda row:row['metadata']['presentation_index'])
            signatures[semantic_state(base['state'])].append(base['metadata']['condition'])
        for same in signatures.values():
            if len(same) > 1:equivalents.append({'split':split,'family':name,'width':width,'conditions':sorted(same)})
        # Width-one first/last aliases are stated separately even when incidental
        # parameter values differ; they are not novel positional semantic cases.
        if width == 1 and name in ('account_status','refund_priority'):
            equivalents.append({'split':split,'family':name,'width':width,
                                'positional_axis':'first_and_last_are_the_same_component',
                                'scope':'Logical position equivalence; exact fact equality is tabulated separately.'})
        essential_checks += essentiality(group,name)
        for kind, n_options in {(row['kind'],len(row['options'])) for row in group}:
            require({row['options'].index(gold(row)) for row in group if row['kind'] == kind and len(row['options']) == n_options} == set(range(n_options)), 'Missing gold option position in parent support')
    expected_parents = {(split,name,width) for split,values in SPLIT_WIDTHS.items() for width in values for name in POLICIES}
    require(parent_keys == expected_parents, 'Missing declared essential parent')
    for split in SPLIT_WIDTHS:
        for field in ('resource','operation','currency'):
            for position in ('first','last'):
                for truth in ('choice','noul/no','noul/yes'):
                    require(scope_cross[split,field,position,truth] > 0, 'Missing joint scope structural cell')
        for field in ('department','topic'):
            for position in ('first','last'):
                for truth in ('choice','noul/no','noul/yes'):
                    require(topic_cross[split,field,position,truth] > 0, 'Missing policy scope structural cell')
        for name in POLICIES:
            if name == 'exact_numeric':continue
            for action in OPTIONS[name]:
                require(actions[split,name,'choice',action] > 0, 'State disposition lacks Choice supervision')
        for width in SPLIT_WIDTHS[split]:
            rank_counts = [money_ranks[split,width,rank] for rank in range(5)]
            require(min(rank_counts) > 0 and max(rank_counts)-min(rank_counts) <= 1,
                    'Six authored balance cases must cover five numeric ranks with balanced counts')
    return {'status':'independent_original_data_oracle_passed', 'model_calls':0, 'schema':schema,
            'target_rows_verified':len(records),'essential_input_interventions':essential_checks,
            'full_typed_presentation_integrity_comparisons':integrity_comparisons,
            'decision_gold_coverage':cells(decisions,['split','family','kind','gold']),
            'state_action_coverage':cells(actions,['split','family','kind','state_action']),
            'condition_kind_truth_coverage':cells(conditions,['split','family','condition','kind_truth']),
            'condition_kind_truth_layout_coverage':cells(layouts,['split','family','condition','kind_truth','layout']),
            'gold_position_coverage':cells(positions,['split','family','kind','option_count','gold_position']),
            'essential_width_coverage':cells(widths,['split','family','width']),
            'signed_numeric_support':cells(numeric,['split','family','quantity','value']),
            'authority_blocked_margin_rows':cells(blocked,['split','family','state_action']),
            'joint_scope_cells':cells(scope_cross,['split','scope_field','required_position','kind_truth']),
            'policy_scope_cells':cells(topic_cross,['split','scope_field','required_position','kind_truth']),
            'money_numeric_rank_case_support':cells(money_ranks,['split','width','numeric_rank']),
            'equivalent_case_annotations':equivalents,
            'record_order_indistinguishable_cases':order_collapses,
            'disclosed_noul_layout_shortfalls':noul_layout_shortfalls,
            'limitations':['Noul yes/no share one layout per authored case; every condition has both Choice layouts, but full condition x Noul-truth x layout crossing is absent within a single parent.',
                           'Only first/last required authority positions are mismatch-probed; middle positions are outcome-essential but not exhaustively mismatch-probed.',
                           'Full presentation integrity uses one bijective identity map on reviewed schema fields plus exact typed facts and declared record layouts. It is a bounded presentation matcher, not arbitrary graph isomorphism. Projected duplicate statistics are separate and do not prove full reference preservation.',
                           'Train widths1/2, Calibration3, Validation4, Test5, OOD6 are a declared composition shift; no IID or natural request claim.',
                           'Dataset oracle and CPU interventions establish authored rules and support, not model capability, natural business success, or official JevBench gain.']}


def audit_directory(directory, contract_path=None):
    root = Path(__file__).resolve().parents[1]
    directory = Path(directory)
    require(directory.name.startswith(VERSION+'-') and not directory.is_symlink(),
            'Only explicitly new original frontier-v8 directories are allowed')
    contract_path = Path(contract_path or root/'reports/frontier-v8-data-20261003/contract.json')
    contract_raw = contract_path.read_bytes()
    contract = json.loads(contract_raw)
    require(contract['version'] == VERSION and contract['split_widths'] == SPLIT_WIDTHS
            and {name:spec['policy'] for name,spec in contract['family_specs'].items()} == POLICIES
            and {name:tuple(spec['conditions']) for name,spec in contract['family_specs'].items()} == CONDITIONS,
            'Reviewed visible contract differs from oracle')
    manifest_path = directory/'manifest.json'
    require(not manifest_path.is_symlink(), 'Manifest symlink rejected')
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    require(manifest['configuration']['generator_version'] == VERSION,
            'Manifest is not a new original v8 dataset; row reads denied')
    expected = {split+'.jsonl' for split in SPLIT_WIDTHS}
    require(set(manifest['files_sha256']) == expected, 'Exactly five declared split files are required')
    records = []
    for split in SPLIT_WIDTHS:
        path = directory/(split+'.jsonl')
        require(not path.is_symlink(), 'Split symlink rejected')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == manifest['files_sha256'][path.name], 'Provisional split checksum differs before parsing')
        parsed = [json.loads(line) for line in raw.splitlines()]
        require(all(row['split'] == split for row in parsed), 'Split-file membership differs')
        records.extend(parsed)
    report = audit_rows(records)
    report['bindings'] = {'contract_sha256':hashlib.sha256(contract_raw).hexdigest(),
                          'manifest_sha256':hashlib.sha256(manifest_raw).hexdigest(),
                          'split_files_sha256':manifest['files_sha256'],
                          'oracle_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'schema_validator_sha256':hashlib.sha256((root/'jev/data.py').read_bytes()).hexdigest()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--contract', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_directory(args.dataset,args.contract)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'rows':report['target_rows_verified'],'model_calls':0}))


if __name__ == '__main__':main()
