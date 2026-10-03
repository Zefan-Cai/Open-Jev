"""Original provisional seven-family compositional controls; CPU only."""
from collections import Counter
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random

from .data import _write_dataset, validate_records

VERSION = "frontier-controls-v8"
WIDTHS = {"train": (1, 2), "calibration": (3,), "validation": (4,), "test": (5,), "ood": (6,)}
POLICIES = {
  "temporal_window": "Start with recorded_delivery_at. Add seconds for each delay and subtract seconds for each advance in delivery_corrections with verified_signature true, issuer_role exactly correction_authority, and case_ref exactly this case_ref. Corrections are distinct by sequence; ignore array order and invalid records. The resulting absolute instant is delivery. An approved exception means review. Otherwise accept from delivery through delivery plus return_window_seconds inclusive; reject outside. Interpret explicit fixed UTC offsets.",
  "exact_numeric": "For balance, sum all credit amount_cents and subtract all debit amount_cents in ledger. For integer_compare, compute those exact signed sums separately for left_ledger and right_ledger, then report below/equal/above. All amounts are nonnegative integer USD cents; use exact integer arithmetic and ignore ledger array order.",
  "joint_capacity": "Each required role needs its greatest-sequence verified signed event with exact request resource/operation/currency scope. Ignore array order, unverified signatures, wrong roles and scopes. Any latest valid revoke means reject revoked consent, even if another role is missing. Otherwise missing roles mean request missing consent. With all latest records grant, execute at/below every capacity; above any capacity means request higher capacity.",
  "latest_authority": "For each required topic, select its greatest valid revision: verified signature from policy_authorities[topic], exact request department in department and credential_scope.department, and exact topic in topic and credential_scope.topic. Resolve all topics before amount/fraud. Missing topic or any latest withdrawn policy means verify policy. Otherwise confirmed_fraud means fraud review. With all latest topics active and no fraud, automatic processing at/below every topic limit; capacity review above any. Ignore array order and invalid records.",
  "account_status": "For each required account, use its greatest-sequence status, ignoring other accounts and array order. No matching event means unknown for that component. Aggregate closed if any component closed; otherwise paused if any paused; otherwise unknown if any missing; otherwise active.",
  "department_route": "Confirmed fraud on any request line means security review before checking policy availability. Otherwise each line requires its greatest valid active signed department policy, verified from policy_authority with exact department and credential_scope. Missing/withdrawn policy or any line amount above its limit means manager review. Approve only when every line has active valid policy and is at/below its own limit.",
  "refund_priority": "Any item's verified recall means recall remediation before all other checks. Otherwise a verified purchase document is required; absent document or any item having neither confirmed defect nor intact seal means reject request. If every item is eligible, sum all item amount_cents: automatic reimbursement at/below automatic_limit_cents; review reimbursement above."
}
CONDITIONS = {
  "temporal_window": [
    "at_start",
    "inside_near",
    "inside_far",
    "at_deadline",
    "before_near",
    "before_far",
    "after_near",
    "after_far",
    "exception_before",
    "exception_inside",
    "exception_deadline",
    "exception_after"
  ],
  "exact_numeric": [
    "balance_negative_near",
    "balance_negative_far",
    "balance_zero_small",
    "balance_zero_large",
    "balance_positive_near",
    "balance_positive_far",
    "compare_below_positive",
    "compare_below_negative",
    "compare_equal_positive",
    "compare_equal_negative",
    "compare_above_positive",
    "compare_above_negative"
  ],
  "joint_capacity": [
    "execute_below_near_first",
    "execute_below_near_last",
    "execute_below_far_first",
    "execute_below_far_last",
    "execute_equal_first",
    "execute_equal_last",
    "execute_regrant_first",
    "execute_regrant_last",
    "scope_first_resource",
    "scope_first_operation",
    "scope_first_currency",
    "scope_last_resource",
    "scope_last_operation",
    "scope_last_currency",
    "missing_absent",
    "missing_invalid",
    "revoke_first_latest",
    "revoke_last_latest",
    "revoke_first_with_missing",
    "revoke_last_with_missing",
    "revoke_first_ignore_invalid",
    "revoke_last_ignore_invalid",
    "revoke_first_over_capacity",
    "revoke_last_over_capacity",
    "capacity_first_near",
    "capacity_last_near",
    "capacity_first_far",
    "capacity_last_far",
    "capacity_first_reduced",
    "capacity_last_reduced",
    "capacity_first_broad",
    "capacity_last_broad"
  ],
  "latest_authority": [
    "automatic_first_near",
    "automatic_last_near",
    "automatic_first_far",
    "automatic_last_far",
    "automatic_first_equal",
    "automatic_last_equal",
    "automatic_first_ignore_invalid",
    "automatic_last_ignore_invalid",
    "capacity_first_near",
    "capacity_last_near",
    "capacity_first_far",
    "capacity_last_far",
    "capacity_first_lowered",
    "capacity_last_lowered",
    "capacity_first_broad",
    "capacity_last_broad",
    "fraud_first_below",
    "fraud_last_below",
    "fraud_first_equal",
    "fraud_last_equal",
    "fraud_first_above",
    "fraud_last_above",
    "fraud_first_broad",
    "fraud_last_broad",
    "verify_first_missing",
    "verify_last_withdrawn",
    "verify_first_signature",
    "verify_last_issuer",
    "verify_first_department",
    "verify_last_department",
    "verify_first_topic",
    "verify_last_topic"
  ],
  "account_status": [
    "active_first",
    "active_last",
    "active_history",
    "paused_first",
    "paused_last",
    "paused_history",
    "closed_first",
    "closed_last",
    "closed_history",
    "unknown_first",
    "unknown_last",
    "unknown_all"
  ],
  "department_route": [
    "approve_below_near",
    "approve_below_far",
    "approve_equal",
    "approve_mixed_below",
    "manager_above_near",
    "manager_above_far",
    "manager_missing",
    "manager_withdrawn",
    "security_below",
    "security_equal",
    "security_above",
    "security_without_policy"
  ],
  "refund_priority": [
    "automatic_below_near",
    "automatic_below_far",
    "automatic_equal",
    "recall_first_without_document",
    "recall_last_over_limit",
    "recall_with_ineligible",
    "reject_document",
    "reject_first_item",
    "reject_last_item",
    "review_above_near",
    "review_above_far",
    "review_broad"
  ]
}
OPTIONS = {
    "temporal_window": ["accept", "reject", "review"],
    "joint_capacity": ["execute", "request missing consent", "reject revoked consent", "request higher capacity"],
    "latest_authority": ["automatic processing", "capacity review", "fraud review", "verify policy"],
    "account_status": ["active", "paused", "closed", "unknown"],
    "department_route": ["approve", "manager review", "security review"],
    "refund_priority": ["automatic reimbursement", "recall remediation", "reject request", "review reimbursement"],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def opaque(prefix, *parts):
    return prefix + '-' + digest(parts)[:20]


def money(cents):
    absolute = abs(cents)
    return f"USD {'-' if cents < 0 else ''}{absolute//100}.{absolute%100:02d}"


def total(entries):
    return sum(e['amount_cents'] * (1 if e['direction'] == 'credit' else -1) for e in entries)


def latest(records, identity, sequence):
    found = {}
    for record in records:
        key = identity(record)
        if key is None:
            continue
        value = record[sequence]
        if (key, value) in found:
            raise ValueError('Ambiguous applicable record')
        found[(key, value)] = record
    return {key: max((r for (k, _), r in found.items() if k == key), key=lambda r: r[sequence])
            for key, _ in found}


def oracle(family, state):
    if family == 'temporal_window':
        applicable = [r for r in state['delivery_corrections'] if r['verified_signature'] is True
                      and r['issuer_role'] == state['correction_authority'] and r['case_ref'] == state['case_ref']]
        if len({r['sequence'] for r in applicable}) != len(applicable):
            raise ValueError('Ambiguous correction sequence')
        corrected = datetime.fromisoformat(state['recorded_delivery_at']) + timedelta(
            seconds=sum(r['seconds'] * (1 if r['direction'] == 'delay' else -1) for r in applicable))
        request = datetime.fromisoformat(state['request_received_at'])
        return 'review' if state['exception_approved'] else 'accept' if corrected <= request <= corrected + timedelta(seconds=state['return_window_seconds']) else 'reject'
    if family == 'exact_numeric':
        if state['task'] == 'balance':
            return money(total(state['ledger']))
        left, right = total(state['left_ledger']), total(state['right_ledger'])
        return 'below' if left < right else 'above' if left > right else 'equal'
    if family == 'joint_capacity':
        request = state['request']; roles = state['trusted_policy']['required_roles']
        scope = {k: request[k] for k in ('resource', 'operation', 'currency')}
        current = latest(state['signed_events'], lambda r: r['issuer_role'] if r['verified_signature'] is True
                         and r['issuer_role'] in roles and r['credential_scope'] == scope else None, 'sequence')
        if any(r['status'] == 'revoke' for r in current.values()):
            return 'reject revoked consent'
        if len(current) != len(roles):
            return 'request missing consent'
        return 'execute' if all(request['amount_cents'] <= r['capacity_cents'] for r in current.values()) else 'request higher capacity'
    if family == 'latest_authority':
        request = state['request']; topics = state['required_topics']
        current = latest(state['signed_policy_registry'], lambda r: r['topic'] if r['topic'] in topics
                         and r['verified_signature'] is True and r['issuer_role'] == state['policy_authorities'][r['topic']]
                         and r['department'] == request['department']
                         and r['credential_scope'] == dict(department=request['department'], topic=r['topic']) else None, 'revision')
        if len(current) != len(topics) or any(r['status'] == 'withdrawn' for r in current.values()):
            return 'verify policy'
        if request['confirmed_fraud']:
            return 'fraud review'
        return 'automatic processing' if all(request['amount_cents'] <= r['automatic_limit_cents'] for r in current.values()) else 'capacity review'
    if family == 'account_status':
        current = latest(state['events'], lambda r: r['account'] if r['account'] in state['required_accounts'] else None, 'sequence')
        statuses = [current[a]['status'] if a in current else 'unknown' for a in state['required_accounts']]
        return next(s for s in ('closed', 'paused', 'unknown', 'active') if s in statuses)
    if family == 'department_route':
        lines = state['request_lines']
        if any(line['confirmed_fraud'] for line in lines):
            return 'security review'
        departments = {line['department'] for line in lines}
        current = latest(state['signed_policies'], lambda r: r['department'] if r['department'] in departments
                         and r['verified_signature'] is True and r['issuer_role'] == state['policy_authority']
                         and r['credential_scope'] == r['department'] else None, 'revision')
        return 'approve' if all(line['department'] in current and current[line['department']]['status'] == 'active'
                                and line['amount_cents'] <= current[line['department']]['approval_limit_cents'] for line in lines) else 'manager review'
    if family == 'refund_priority':
        items = state['items']
        if any(item['verified_recall'] for item in items):
            return 'recall remediation'
        if not state['purchase_document_verified'] or any(not (item['defect_confirmed'] or item['seal_intact']) for item in items):
            return 'reject request'
        return 'automatic reimbursement' if sum(i['amount_cents'] for i in items) <= state['trusted_policy']['automatic_limit_cents'] else 'review reimbursement'
    raise ValueError('Unknown family')


def partition(rng, amount, count):
    if amount < count:
        raise ValueError('Positive operands required')
    cuts = [0] + sorted(rng.sample(range(1, amount), count-1)) + [amount]
    return [b-a for a, b in zip(cuts, cuts[1:])]


def ledger(rng, width, result, entity):
    credits = abs(result) + width*5000 + rng.randrange(10000, 50000)
    positive = partition(rng, credits, width)
    for _ in range(1000):
        negative = partition(rng, credits-result, width)
        if width == 1 or not set(positive).intersection(negative):
            return [dict(entry_ref=opaque('entry', entity, side, i), direction=side, amount_cents=value)
                    for side, values in (('credit', positive), ('debit', negative)) for i, value in enumerate(values)]
    raise ValueError('Could not construct nonpadding operands')


def case(family, width, index, rng, entity):
    limit = rng.randrange(10000, 50000)*rng.choice((1, 10, 100))
    policy = POLICIES[family]; condition = CONDITIONS[family][index]; axis = {}
    if family == 'temporal_window':
        start = datetime(2027, rng.randrange(1, 13), rng.randrange(10, 25), rng.randrange(24), tzinfo=timezone.utc)
        authority = opaque('authority', entity)
        corrections = [dict(sequence=i+1, issuer_role=authority, verified_signature=True, case_ref=entity,
                            direction='delay' if i % 2 == 0 else 'advance', seconds=rng.randrange(1, 1001)) for i in range(width)]
        corrected = start + timedelta(seconds=sum(r['seconds'] * (1 if r['direction'] == 'delay' else -1) for r in corrections))
        window = rng.choice((3601, 86403, 172801)); far = window+60
        delta = (0, 1, window//2, window, -1, -far, window+1, window+far, -far, window//2, window, window+far)[index]
        zone = timezone(timedelta(minutes=rng.choice((-420, 0, 330))))
        state = dict(case_ref=entity, recorded_delivery_at=start.isoformat(), correction_authority=authority,
                     delivery_corrections=corrections, request_received_at=(corrected+timedelta(seconds=delta)).astimezone(zone).isoformat(),
                     return_window_seconds=window, exception_approved=index >= 8, policy=policy)
    elif family == 'exact_numeric':
        state = dict(case_ref=entity, task='balance' if index < 6 else 'integer_compare', currency='USD', policy=policy)
        if index < 6:
            result = (-1, -limit//2, 0, 0, 1, limit//2)[index]
            state['ledger'] = ledger(rng, width, result, entity)
        else:
            right = limit if index % 2 == 0 else -limit
            gap = rng.choice((1, 99, 1001))
            left = right + (-gap if index < 8 else 0 if index < 10 else gap)
            state.update(left_ledger=ledger(rng, width, left, entity+'/left'), right_ledger=ledger(rng, width, right, entity+'/right'))
    elif family == 'joint_capacity':
        roles = [opaque('authority', entity, i) for i in range(width+1)]
        selected = 0 if 'first' in condition or condition == 'missing_absent' else len(roles)-1
        request = dict(resource=opaque('resource', entity), operation='release', currency='USD', amount_cents=limit)
        scope = {key: request[key] for key in ('resource', 'operation', 'currency')}
        events = [dict(issuer_role=role, verified_signature=True, credential_scope=copy.deepcopy(scope), sequence=1,
                       status='grant', capacity_cents=limit if i == selected else limit+200) for i, role in enumerate(roles)]
        event = events[selected]; axis['affected_role_index'] = selected
        if index < 8:
            request['amount_cents'] = limit-(1 if index < 2 else 1000 if index < 4 else 0)
            if index >= 6:
                events.extend([{**copy.deepcopy(event), 'sequence': 2, 'status': 'revoke'}, {**copy.deepcopy(event), 'sequence': 3}])
        elif index < 16:
            if index < 14:
                field = ('resource', 'operation', 'currency')[(index-8) % 3]
                event['credential_scope'][field] += '-different'; axis['scope_field'] = field
            elif index == 14:
                events.remove(event)
            else:
                event['verified_signature'] = False
        elif index < 24:
            events.append({**copy.deepcopy(event), 'sequence': 2, 'status': 'revoke'})
            if index in (18, 19):
                events.remove(events[len(roles)-1 if selected == 0 else 0])
            if index in (20, 21):
                events.append({**copy.deepcopy(event), 'sequence': 99, 'verified_signature': False})
            if index >= 22:
                request['amount_cents'] = limit+1000
        else:
            request['amount_cents'] = limit+(1 if index < 26 else 1000 if index < 28 else 0 if index < 30 else limit//2)
            if index in (28, 29):
                events.append({**copy.deepcopy(event), 'sequence': 2, 'capacity_cents': limit-100})
        state = dict(case_ref=entity, request=request, trusted_policy=dict(required_roles=roles, rules=policy), signed_events=events)
    elif family == 'latest_authority':
        topics = [opaque('topic', entity, i) for i in range(width+1)]
        authorities = {topic: opaque('authority', entity, topic) for topic in topics}
        selected = 0 if 'first' in condition else len(topics)-1; topic = topics[selected]
        department = opaque('department', entity)
        policies = [dict(topic=t, department=department, credential_scope=dict(department=department, topic=t),
                         issuer_role=authorities[t], verified_signature=True, revision=1, status='active',
                         automatic_limit_cents=limit if i == selected else limit+200) for i, t in enumerate(topics)]
        current = policies[selected]; request = dict(department=department, amount_cents=limit, confirmed_fraud=False)
        axis['affected_topic_index'] = selected
        if index < 8:
            request['amount_cents'] -= 1 if index < 2 else 1000 if index < 4 else 0
            if index >= 6:
                policies.append({**copy.deepcopy(current), 'revision': 99, 'verified_signature': False, 'automatic_limit_cents': 0})
        elif index < 16:
            request['amount_cents'] += 1 if index < 10 else 1000 if index < 12 else 0 if index < 14 else limit//2
            if index in (12, 13):
                policies.append({**copy.deepcopy(current), 'revision': 2, 'automatic_limit_cents': limit-100})
        elif index < 24:
            request['confirmed_fraud'] = True
            request['amount_cents'] += -1000 if index < 18 else 0 if index < 20 else 1 if index < 22 else limit//2
        else:
            request['confirmed_fraud'] = index % 2 == 0
            if index == 24:
                policies.remove(current)
            elif index == 25:
                current['status'] = 'withdrawn'
            elif index == 26:
                current['verified_signature'] = False
            elif index == 27:
                current['issuer_role'] += '-different'
            else:
                field = 'department' if index < 30 else 'topic'
                current['credential_scope'][field] += '-different'; axis['scope_field'] = field
        state = dict(case_ref=entity, request=request, required_topics=topics, policy_authorities=authorities,
                     signed_policy_registry=policies, trust_contract=policy)
    elif family == 'account_status':
        accounts = [opaque('account', entity, i) for i in range(width)]
        sequence = rng.randrange(100, 10000)
        events = [dict(account=a, sequence=sequence, status='active') for a in accounts]
        selected = 0 if index % 3 == 0 else len(accounts)-1
        status = ('active', 'paused', 'closed', 'unknown')[index//3]
        if status == 'unknown':
            events = [] if index == 11 else [e for e in events if e['account'] != accounts[selected]]
        else:
            events[selected]['status'] = status
            if index % 3 == 2:
                events.append(dict(account=accounts[selected], sequence=sequence-1, status='closed' if status != 'closed' else 'paused'))
        state = dict(case_ref=entity, required_accounts=accounts, events=events, rule=policy)
    elif family == 'department_route':
        departments = [opaque('department', entity, i) for i in range(width)]
        issuer = opaque('authority', entity); selected = index % width
        lines = [dict(department=d, amount_cents=limit-200, confirmed_fraud=False) for d in departments]
        policies = [dict(department=d, credential_scope=d, issuer_role=issuer, verified_signature=True,
                         revision=1, status='active', approval_limit_cents=limit) for d in departments]
        line = lines[selected]
        if index < 4:
            line['amount_cents'] = limit-(1 if index == 0 else 1000 if index == 1 else 0 if index == 2 else 200)
        elif index < 8:
            if index < 6:
                line['amount_cents'] = limit+(1 if index == 4 else 1000)
            elif index == 6:
                policies.pop(selected)
            else:
                policies[selected]['status'] = 'withdrawn'
        else:
            line['confirmed_fraud'] = True
            line['amount_cents'] = limit+(-1000 if index == 8 else 0 if index == 9 else 1000)
            if index == 11:
                policies = []
        state = dict(case_ref=entity, request_lines=lines, policy_authority=issuer, signed_policies=policies, policy=policy)
    elif family == 'refund_priority':
        amount = limit+(-1 if index == 0 else -1000 if index == 1 else 0 if index < 9 else 1 if index == 9 else 1000 if index == 10 else limit//2)
        if index == 4:
            amount = limit+1000
        items = [dict(item_ref=opaque('item', entity, i), amount_cents=value, defect_confirmed=True,
                      seal_intact=False, verified_recall=False) for i, value in enumerate(partition(rng, amount, width))]
        verified = True
        if 3 <= index < 6:
            items[-1 if index == 4 else 0]['verified_recall'] = True
            verified = index != 3
            if index == 5:
                items[-1]['defect_confirmed'] = False
        elif index < 9 and index >= 6:
            if index == 6:
                verified = False
            else:
                items[0 if index == 7 else -1]['defect_confirmed'] = False
        state = dict(case_ref=entity, purchase_document_verified=verified, items=items,
                     trusted_policy=dict(currency='USD', automatic_limit_cents=limit, ordered_rule=policy))
    else:
        raise ValueError('Unknown family')
    return state, axis


def presentation(state, token, reverse):
    names = {}
    prefixes = ('case-', 'authority-', 'topic-', 'account-', 'department-', 'resource-', 'item-', 'entry-')

    def rename(value):
        if isinstance(value, str) and value.startswith(prefixes):
            return names.setdefault(value, opaque(value.split('-')[0], token, value))
        return value

    def transform(value, key=None):
        if isinstance(value, dict):
            pairs = list(value.items())
            if reverse:
                pairs.reverse()
            return {rename(k): transform(v, k) for k, v in pairs}
        if isinstance(value, list):
            values = [transform(v) for v in value]
            return list(reversed(values)) if reverse and key not in ('required_roles', 'required_topics', 'required_accounts') else values
        return rename(value)

    return transform(state)


def records(seed=20261003):
    positions = Counter(); numeric_ranks = Counter()
    rank_schedules = {}
    for split, widths in WIDTHS.items():
        for ordinal, width in enumerate(widths):
            schedule = list(range(5)) + [ordinal % 5]
            random.Random(digest([seed, split, width, 'amount-rank-schedule'])).shuffle(schedule)
            rank_schedules[(split, width)] = schedule
    for split, widths in WIDTHS.items():
        for family in CONDITIONS:
            for width in widths:
                group = f'{VERSION}/{seed}/{split}/{family}/{width}'
                for index, condition in enumerate(CONDITIONS[family]):
                    rng = random.Random(digest([seed, group, index]))
                    entity = opaque('case', seed, group, index)
                    state, axes = case(family, width, index, rng, entity)
                    answer = oracle(family, state)
                    task = state.get('task', family)
                    if family == 'exact_numeric' and task == 'balance':
                        result = total(state['ledger']); rank = rank_schedules[(split, width)][numeric_ranks[(split, family, width)]]
                        numeric_ranks[(split, family, width)] += 1
                        step = rng.choice((1, 7, 99))
                        choices = [money(result+(j-rank)*step) for j in range(5)]
                        axes = {**axes, 'money_gold_numeric_rank': rank}
                    else:
                        choices = ['below', 'equal', 'above'] if family == 'exact_numeric' else list(OPTIONS[family])
                    # Both Noul truths use the same layout, so order cannot encode yes/no.
                    noul_reverse = (index+width) % 2 == 1
                    wrong = [value for value in choices if value != answer]
                    if family == 'exact_numeric' and task == 'balance':
                        below = choices[:rank]; above = choices[rank+1:]
                        side = below if index % 2 == 0 else above
                        wrong = side or wrong
                    proposed_false = wrong[rng.randrange(len(wrong))]
                    for variant in range(4):
                        kind = 'choice' if variant < 2 else 'noul'
                        reverse = variant == 1 if kind == 'choice' else noul_reverse
                        current = presentation(state, [seed, group, index, variant], reverse)
                        proposed = None
                        if kind == 'choice':
                            options = [option for option in choices if option != answer]
                            rng.shuffle(options)
                            position = positions[(split, family, task)] % len(choices)
                            positions[(split, family, task)] += 1
                            options.insert(position, answer)
                            question = ('Apply the supplied rule to these recorded facts. Which outcome follows?',
                                        'Use the authoritative rule and facts to select the resulting disposition.')[variant]
                            target = [float(option == answer) for option in options]
                        else:
                            proposed = answer if variant == 2 else proposed_false
                            options = ['no', 'yes']; target = [float(variant == 3), float(variant == 2)]
                            question = f"Does the supplied rule establish '{proposed}' for this case?"
                        yield dict(id=group+f'/c{index}/v{variant}', group_id=group, split=split, source=VERSION,
                                   state=current, question=question, kind=kind, options=options, target=target,
                                   metadata=dict(family='routing' if family == 'account_status' else 'policy', scenario_family=family,
                                                 condition=condition, composition_width=width, case_index=index,
                                                 presentation_index=variant, layout='reverse' if reverse else 'forward', **axes,
                                                 entity_ids=[group], template_id=VERSION+'/'+family+('/ood' if split == 'ood' else '/id'),
                                                 proposed_outcome=proposed, target_basis='authored_exact_state_rule',
                                                 provenance=dict(type='synthetic', license='CC0-1.0', generator_version=VERSION,
                                                                 seed=seed, group_index=width, variant=index*4+variant,
                                                                 split_policy='whole_essential_composition_parent', upstream_rows_imported=0)))


def build(output, seed=20261003):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a fresh provisional output; existing data cannot be overwritten')
    rows = list(records(seed)); schema = validate_records(rows)
    output.mkdir(parents=True, exist_ok=False)
    manifest = _write_dataset(rows, output, dict(type='synthetic', status='provisional_not_frozen',
        generator_version=VERSION, seed=seed, license='CC0-1.0', upstream_rows_imported=0,
        composition_widths=WIDTHS, model_calls=0, training_runs=0,
        structural_shift='Train widths1/2; Calibration3/Validation4/Test5/OOD6. Controlled unseen composition, not IID or natural requests.',
        limitations='First/last role/topic scope crossings only; middle mismatches nonexhaustive. Repeated logical cases or atomic motifs are not distinct independent scenarios. Independent oracle/isolation approval pending.'))
    manifest.update(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    status='provisional_not_frozen', oracle_review='pending_independent', isolation_review='pending_independent',
                    condition_kind_truth=dict(Counter(r['metadata']['scenario_family']+'/'+r['metadata']['condition']+'/'+r['kind']+
                                              ('/'+r['options'][r['target'].index(1.)] if r['kind'] == 'noul' else '') for r in rows)),
                    schema_unique_inputs=schema['unique_inputs'], duplicate_schema_input_rows=len(rows)-schema['unique_inputs'],
                    distinctness_note='Schema fingerprints are not proof of distinct semantic cases; independent identity/magnitude/order abstraction review remains pending.')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return manifest
