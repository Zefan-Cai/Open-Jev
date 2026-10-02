"""Independent CPU replay of original boundary controls; no model calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from jev.data import SPLITS, validate_records

VERSION = 'boundary-controls-v7'
GROUP_COUNTS = dict(train=32,calibration=4,validation=4,test=8,ood=8)
CONDITIONS = {
    'temporal_window':('before','at_start','inside','at_deadline','after','before_exception','after_exception','equivalent_outside'),
    'exact_numeric':('balance_negative','balance_zero','balance_positive','balance_cancellation','compare_below','compare_equal','compare_above','compare_negative'),
    'joint_capacity':('equality','latest_regrant','role_absent','scope_mismatch','latest_revoke','revoke_ignores_invalid_grant','one_cent_above','latest_reduced_capacity'),
    'latest_authority':('equality','ignore_invalid_high_revision','one_cent_above','latest_lower_limit','fraud_with_active','fraud_over_capacity','latest_withdrawn','missing_authority'),
}
# Independently reviewed authored contracts. Arbitrary prose is not an oracle.
CONTRACTS = {
    'temporal_window':"An approved exception routes the case to review, including a request outside the window. Without an approved exception, accept exactly when the request instant is from delivery through delivery plus return_window_seconds, including both endpoints; otherwise reject. Use the stated fixed UTC offsets to compare absolute instants, rather than displayed clocks.",
    'exact_numeric':"For task balance, add all credit amount_cents and subtract all debit amount_cents. Each amount is a nonnegative integer count of USD cents; 100 cents is one USD. The order of entries does not affect the balance. For task integer_compare, compare the signed integer left_cents to right_cents: below, equal, or above.",
    'joint_capacity':"Each required role needs a verified signed event whose credential_scope exactly matches the requested resource, operation and currency. For each role use its greatest valid sequence, regardless of list order; unverified, wrong-role or wrong-scope events have no authority. Any latest valid revoke means reject revoked consent, even if another role is absent. Otherwise an absent role means request missing consent. With all latest events grant, request higher capacity if amount_cents exceeds either capacity_cents; equality means execute.",
    'latest_authority':"Resolve authority before checking fraud or amount. A valid policy needs a verified signature from policy_authority and exact request department in both department and credential_scope. Use its greatest valid revision, ignoring array order and all invalid records. No valid policy or a latest withdrawn policy means verify policy; never fall back to an older active policy. With the latest policy active, confirmed_fraud means fraud review regardless of amount. Otherwise amount_cents at or below its automatic_limit_cents means automatic processing, and an amount above that limit means capacity review.",
}
CHOICE_OPTIONS = {
    'temporal_window':{'accept','reject','review'},
    'joint_capacity':{'execute','request missing consent','reject revoked consent','request higher capacity'},
    'latest_authority':{'automatic processing','capacity review','fraud review','verify policy'},
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def integer(value, name, *, nonnegative=False):
    if type(value) is not int or (nonnegative and value < 0):
        raise ValueError(name+' must be an exact '+('nonnegative ' if nonnegative else '')+'integer')
    return value


def instant_seconds(value):
    """Gregorian ordinal arithmetic, independent of datetime interval code."""
    match = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(Z|[+-]\d{2}:\d{2})',value)
    if not match:raise ValueError('Explicit whole-second timestamp with UTC offset required')
    year,month,day,hour,minute,second = map(int,match.groups()[:6])
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days = (31,29 if leap else 28,31,30,31,30,31,31,30,31,30,31)
    if not (1 <= year <= 9999 and 1 <= month <= 12 and 1 <= day <= days[month-1]
            and 0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise ValueError('Invalid calendar timestamp')
    zone = match.group(7);offset = 0
    if zone != 'Z':
        zh,zm = int(zone[1:3]),int(zone[4:6])
        if zh >= 24 or zm >= 60:raise ValueError('Invalid UTC offset')
        offset = (1 if zone[0] == '+' else -1)*(zh*3600+zm*60)
    prior = year-1
    ordinal = prior*365+prior//4-prior//100+prior//400+sum(days[:month-1])+day-1
    return ordinal*86400+hour*3600+minute*60+second-offset


def ledger_total(entries):
    total = 0
    for entry in entries:
        cents = integer(entry['amount_cents'],'ledger cents',nonnegative=True)
        if entry['direction'] not in ('credit','debit'):raise ValueError('Unknown ledger entry type')
        total += cents if entry['direction'] == 'credit' else -cents
    return total


def cents_label(cents,currency):
    integer(cents,'balance')
    absolute = abs(cents)
    return f"{currency} {'-' if cents < 0 else ''}{absolute//100}.{absolute%100:02d}"


def latest_credentials(state):
    request = state['request'];roles = state['trusted_policy']['required_roles']
    if not isinstance(roles,list) or len(roles) != 2 or len(set(roles)) != 2:raise ValueError('Two distinct required roles are required')
    scope = {key:request[key] for key in ('resource','operation','currency')}
    latest,seen = {},set()
    for event in sorted(state['signed_events'],key=lambda e:integer(e['sequence'],'signed sequence',nonnegative=True)):
        if type(event['verified_signature']) is not bool:raise ValueError('Signature flag must be boolean')
        if (event['issuer_role'] not in roles or event['verified_signature'] is not True
                or event['credential_scope'] != scope):continue
        identity = (event['issuer_role'],event['sequence'])
        if identity in seen:raise ValueError('Conflicting signed sequence for one authority')
        seen.add(identity)
        if event['status'] not in ('grant','revoke'):raise ValueError('Unknown credential status')
        integer(event['capacity_cents'],'signed capacity',nonnegative=True)
        latest[event['issuer_role']] = event
    return latest


def latest_policy(state):
    request = state['request'];policies=[];seen=set()
    for policy in state['signed_policy_registry']:
        if type(policy['verified_signature']) is not bool:raise ValueError('Signature flag must be boolean')
        if (policy['verified_signature'] is not True or policy['issuer_role'] != state['policy_authority']
                or policy['department'] != request['department'] or policy['credential_scope'] != request['department']):continue
        revision = integer(policy['revision'],'signed policy revision',nonnegative=True)
        if revision in seen:raise ValueError('Conflicting signed policy revision')
        seen.add(revision)
        if policy['status'] not in ('active','withdrawn'):raise ValueError('Unknown signed policy status')
        integer(policy['automatic_limit_cents'],'signed limit',nonnegative=True)
        policies.append(policy)
    return sorted(policies,key=lambda p:p['revision'],reverse=True)[0] if policies else None


def temporal_outcome(state):
    window = integer(state['return_window_seconds'],'window seconds')
    if window <= 0 or type(state['exception_approved']) is not bool:raise ValueError('Invalid authored interval contract')
    elapsed = instant_seconds(state['request_received_at'])-instant_seconds(state['delivered_at'])
    return 'review' if state['exception_approved'] else 'accept' if 0 <= elapsed <= window else 'reject'


def joint_outcome(state):
    amount = integer(state['request']['amount_cents'],'request cents',nonnegative=True)
    latest = latest_credentials(state)
    if any(event['status'] == 'revoke' for event in latest.values()):return 'reject revoked consent'
    if set(latest) != set(state['trusted_policy']['required_roles']):return 'request missing consent'
    if any(event['capacity_cents'] < amount for event in latest.values()):return 'request higher capacity'
    return 'execute'


def authority_outcome(state):
    request = state['request'];amount = integer(request['amount_cents'],'request cents',nonnegative=True)
    if type(request['confirmed_fraud']) is not bool:raise ValueError('Fraud flag must be boolean')
    policy = latest_policy(state)
    if not policy or policy['status'] == 'withdrawn':return 'verify policy'
    if request['confirmed_fraud']:return 'fraud review'
    return 'automatic processing' if amount <= policy['automatic_limit_cents'] else 'capacity review'


def numeric_outcome(state):
    if state['task'] == 'balance':
        if state['currency'] != 'USD':raise ValueError('The authored exact ledger contract uses USD cents')
        return cents_label(ledger_total(state['ledger']),state['currency'])
    if state['task'] != 'integer_compare':raise ValueError('Unknown exact numeric task')
    left,right = integer(state['left_cents'],'left cents'),integer(state['right_cents'],'right cents')
    return 'below' if left < right else 'above' if left > right else 'equal'


def replay(row):
    family,state = row['metadata']['scenario_family'],row['state']
    if family not in CONDITIONS:raise ValueError('Unknown original boundary family')
    contract = state['trusted_policy']['rules'] if family == 'joint_capacity' else state['trust_contract'] if family == 'latest_authority' else state['policy']
    if contract != CONTRACTS[family]:raise ValueError('Visible authored rule differs from the reviewed contract')
    outcome = {'temporal_window':temporal_outcome,'exact_numeric':numeric_outcome,
               'joint_capacity':joint_outcome,'latest_authority':authority_outcome}[family](state)
    if row['kind'] == 'noul':
        proposal = re.fullmatch(r"Does the supplied rule establish '([^']+)' for this case\?",row['question'])
        if not proposal or row['metadata']['proposed_outcome'] != proposal.group(1):
            raise ValueError('Visible Noul proposal differs from declared metadata')
        supported = CHOICE_OPTIONS.get(family,{'below','equal','above'})
        if family == 'exact_numeric' and state['task'] == 'balance':
            if not re.fullmatch(r'USD -?\d+\.\d{2}',proposal.group(1)):raise ValueError('Unknown exact balance proposition')
        elif proposal.group(1) not in supported:raise ValueError('Unknown authored action proposition')
        expected = [float(proposal.group(1) != outcome),float(proposal.group(1) == outcome)]
        if row['options'] != ['no','yes']:raise ValueError('Noul label order differs')
    elif row['kind'] == 'choice':
        if row['question'] != 'Apply the supplied exact rule to the recorded facts. Which stated outcome follows?':
            raise ValueError('Visible Choice instruction differs from the reviewed contract')
        if row['metadata']['proposed_outcome'] is not None:raise ValueError('Choice must not carry a Noul proposal')
        candidates = CHOICE_OPTIONS.get(family,{'below','equal','above'})
        if family == 'exact_numeric' and state['task'] == 'balance':
            if state['currency'] != 'USD' or len(row['options']) != 5 or any(not re.fullmatch(r'USD -?\d+\.\d{2}',v) for v in row['options']):
                raise ValueError('Exact balance candidates must be five integer-cent USD labels')
        elif set(row['options']) != candidates:raise ValueError('Choice candidates differ from the authored action set')
        if outcome not in row['options']:raise ValueError('Exact outcome is absent from Choice candidates')
        expected = [float(option == outcome) for option in row['options']]
    else:raise ValueError('Boundary controls support only Choice and Noul')
    if row['target'] != expected:raise ValueError('Independent fact replay disagrees with target: '+row['id'])
    return outcome


def valid_events(state):
    scope = {key:state['request'][key] for key in ('resource','operation','currency')}
    return [event for event in state['signed_events'] if event['verified_signature'] is True
            and event['issuer_role'] in state['trusted_policy']['required_roles'] and event['credential_scope'] == scope]


def valid_policies(state):
    department = state['request']['department']
    return [policy for policy in state['signed_policy_registry'] if policy['verified_signature'] is True
        and policy['issuer_role'] == state['policy_authority'] and policy['department'] == department
        and policy['credential_scope'] == department]


def check_conditions(group):
    family = group[0]['metadata']['scenario_family']
    cases = {row['metadata']['condition']:row for row in group}
    if len(group) != 8 or set(cases) != set(CONDITIONS[family]):
        raise ValueError('Every group requires eight distinct authored conditions')
    for condition,row in cases.items():
        state = row['state'];outcome = replay(row);valid = False
        if family == 'temporal_window':
            delta = instant_seconds(state['request_received_at'])-instant_seconds(state['delivered_at'])
            duration = state['return_window_seconds'];exception = state['exception_approved']
            valid = {'before':delta == -1 and not exception,'at_start':delta == 0 and not exception,
                'inside':0 < delta < duration and not exception,'at_deadline':delta == duration and not exception,
                'after':delta == duration+1 and not exception,'before_exception':delta == -1 and exception,
                'after_exception':delta == duration+1 and exception}.get(condition,False)
            reference = cases['before' if delta < 0 else 'after']['state']
            if condition == 'equivalent_outside':
                valid = (not exception and outcome == 'reject' and state['case_ref'] == reference['case_ref']
                    and state['delivered_at'] == reference['delivered_at'] and duration == reference['return_window_seconds']
                    and instant_seconds(state['request_received_at']) == instant_seconds(reference['request_received_at'])
                    and state['request_received_at'][-6:] != reference['request_received_at'][-6:])
        elif family == 'exact_numeric':
            if condition.startswith('balance_') and state['task'] == 'balance':
                total = ledger_total(state['ledger'])
                valid = {'balance_negative':total < 0,'balance_zero':total == 0,'balance_positive':total > 0}.get(condition,False)
                if condition == 'balance_cancellation':
                    base = cases['balance_positive']['state']
                    pairs = {(entry['direction'],entry['amount_cents']) for entry in state['ledger']}
                    valid = (total == ledger_total(base['ledger']) and len(state['ledger']) >= len(base['ledger'])+2
                        and any(direction == 'credit' and amount > 0 and ('debit',amount) in pairs for direction,amount in pairs))
            elif condition.startswith('compare_') and state['task'] == 'integer_compare':
                left,right = state['left_cents'],state['right_cents']
                valid = {'compare_below':left == right-1,'compare_equal':left == right,'compare_above':left == right+1,
                    'compare_negative':left < 0 and right < 0 and abs(left-right) == 1}.get(condition,False)
        elif family == 'joint_capacity':
            events = valid_events(state);latest = latest_credentials(state);amount = state['request']['amount_cents']
            minimum = min((e['capacity_cents'] for e in latest.values()),default=None)
            if condition == 'equality':valid = outcome == 'execute' and amount == minimum
            elif condition == 'latest_regrant':
                valid = outcome == 'execute' and any(e['status'] == 'revoke' and e['issuer_role'] == last['issuer_role']
                    and e['sequence'] < last['sequence'] for last in latest.values() for e in events)
            elif condition == 'role_absent':
                base = valid_events(cases['equality']['state'])
                valid = outcome == 'request missing consent' and any(not any(e['issuer_role'] == original['issuer_role']
                    and e['sequence'] == original['sequence'] for e in state['signed_events']) for original in base)
            elif condition == 'scope_mismatch':
                scope = {key:state['request'][key] for key in ('resource','operation','currency')}
                valid = outcome == 'request missing consent' and any(e['verified_signature'] is True
                    and e['issuer_role'] in state['trusted_policy']['required_roles'] and set(e['credential_scope']) == set(scope)
                    and sum(e['credential_scope'][key] != value for key,value in scope.items()) == 1 for e in state['signed_events'])
            elif condition == 'latest_revoke':valid = outcome == 'reject revoked consent'
            elif condition == 'revoke_ignores_invalid_grant':
                valid = outcome == 'reject revoked consent' and any(last['status'] == 'revoke'
                    and event['issuer_role'] == last['issuer_role'] and event['sequence'] > last['sequence']
                    and event['status'] == 'grant' and event not in events for last in latest.values() for event in state['signed_events'])
            elif condition == 'one_cent_above':valid = outcome == 'request higher capacity' and amount == minimum+1
            elif condition == 'latest_reduced_capacity':
                valid = outcome == 'request higher capacity' and amount == minimum+1 and any(
                    earlier['issuer_role'] == last['issuer_role'] and earlier['sequence'] < last['sequence']
                    and earlier['status'] == 'grant' and earlier['capacity_cents'] >= amount
                    and last['capacity_cents'] == amount-1 for earlier in events for last in latest.values())
        else:
            policy = latest_policy(state);valid_registry = valid_policies(state)
            amount = state['request']['amount_cents'];fraud = state['request']['confirmed_fraud']
            limit = policy['automatic_limit_cents'] if policy else None
            if condition == 'equality':valid = outcome == 'automatic processing' and amount == limit
            elif condition == 'ignore_invalid_high_revision':
                valid = outcome == 'automatic processing' and amount == limit and any(p not in valid_registry
                    and p['revision'] > policy['revision'] for p in state['signed_policy_registry'])
            elif condition == 'one_cent_above':valid = outcome == 'capacity review' and amount == limit+1
            elif condition == 'latest_lower_limit':
                valid = outcome == 'capacity review' and amount == limit+1 and any(p['revision'] < policy['revision']
                    and p['status'] == 'active' and p['automatic_limit_cents'] == amount for p in valid_registry)
            elif condition == 'fraud_with_active':valid = outcome == 'fraud review' and fraud and amount == limit
            elif condition == 'fraud_over_capacity':valid = outcome == 'fraud review' and fraud and amount == limit+1
            elif condition == 'latest_withdrawn':
                valid = (outcome == 'verify policy' and policy is not None and policy['status'] == 'withdrawn'
                    and any(p['status'] == 'active' and p['revision'] < policy['revision'] for p in valid_registry))
            elif condition == 'missing_authority':valid = outcome == 'verify policy' and policy is None and fraud
        if not valid:raise ValueError('Condition label disagrees with independently checked facts: '+row['id']+'/'+condition)


def audit_rows(records):
    schema = validate_records(records)
    groups = defaultdict(list);coverage = defaultdict(Counter);positions = defaultdict(Counter)
    family_counts = Counter();noul_targets = defaultdict(Counter);outcomes = defaultdict(Counter);structures = defaultdict(Counter)
    for row in records:
        metadata = row['metadata'];provenance = metadata['provenance'];family = metadata['scenario_family']
        if row['source'] != VERSION or provenance['generator_version'] != VERSION or provenance.get('upstream_rows_imported') != 0:
            raise ValueError('Wrong original-only boundary provenance')
        outcome = replay(row);condition = metadata['condition'];split = row['split']
        if condition not in CONDITIONS[family]:raise ValueError('Unknown authored condition')
        groups[row['group_id']].append(row);family_counts[(split,family)] += 1;outcomes[(split,family)][outcome] += 1
        key = row['kind'] if row['kind'] == 'choice' else 'noul/'+row['options'][row['target'].index(1.)]
        coverage[(split,family,condition)][key] += 1
        if row['kind'] == 'choice':positions[(split,family,len(row['options']))][row['target'].index(1.)] += 1
        else:noul_targets[(split,family)][row['target'].index(1.)] += 1
    for group in groups.values():
        if (len(group) != 8 or {r['metadata']['provenance']['variant'] for r in group} != set(range(8))
                or len({(r['split'],r['metadata']['scenario_family']) for r in group}) != 1):
            raise ValueError('Incomplete or mixed eight-variant group')
        check_conditions(group)
        family = group[0]['metadata']['scenario_family'];cases = {r['metadata']['condition']:r for r in group}
        for condition,row in cases.items():
            state = row['state'];axis = None
            if family == 'temporal_window' and condition == 'equivalent_outside':
                axis = 'before' if instant_seconds(state['request_received_at']) < instant_seconds(state['delivered_at']) else 'after'
            elif family == 'exact_numeric' and condition == 'compare_negative':axis = numeric_outcome(state)
            elif family == 'joint_capacity' and condition in ('latest_regrant','role_absent','scope_mismatch','latest_revoke','revoke_ignores_invalid_grant','latest_reduced_capacity'):
                roles = state['trusted_policy']['required_roles'];latest = latest_credentials(state)
                if condition in ('role_absent','scope_mismatch'):changed = [role for role in roles if role not in latest]
                elif condition in ('latest_revoke','revoke_ignores_invalid_grant'):changed = [role for role in roles if latest[role]['status'] == 'revoke']
                elif condition == 'latest_reduced_capacity':changed = [role for role in roles if latest[role]['capacity_cents'] < state['request']['amount_cents']]
                else:
                    base = latest_credentials(cases['equality']['state'])
                    changed = [role for role in roles if latest[role]['sequence'] > base[role]['sequence']]
                if len(changed) != 1:raise ValueError('Joint structural role axis is not uniquely defined')
                axis = 'required_role_'+str(roles.index(changed[0]))
            if axis:
                key = row['kind'] if row['kind'] == 'choice' else 'noul/'+row['options'][row['target'].index(1.)]
                structures[(row['split'],family,condition,axis)][key] += 1
    for split in SPLITS:
        for family,conditions in CONDITIONS.items():
            n = GROUP_COUNTS[split]
            if family_counts[(split,family)] != n*8:raise ValueError('Fixed family/split denominators differ')
            if noul_targets[(split,family)] != Counter({0:n*2,1:n*2}):raise ValueError('Noul truth labels are imbalanced')
            for condition in conditions:
                if coverage[(split,family,condition)] != Counter({'choice':n//2,'noul/no':n//4,'noul/yes':n//4}):
                    raise ValueError('Condition lacks balanced Choice/Noul yes/no coverage')
    for (split,family,classes),counts in positions.items():
        values = [counts[index] for index in range(classes)]
        if max(values)-min(values) > 1:raise ValueError('Choice target positions are imbalanced')
    for (split,family,condition,axis),counts in structures.items():
        required = ('choice','noul/no','noul/yes') if split in ('train','test','ood') else ('choice',)
        if any(counts[key] == 0 for key in required) or not counts['noul/no']+counts['noul/yes']:
            raise ValueError('State-derived structural axis is coupled to kind/truth')
    axes = defaultdict(set)
    for split,family,condition,axis in structures:axes[(split,family,condition)].add(axis)
    for (split,family,condition),values in axes.items():
        expected = {'before','after'} if family == 'temporal_window' else {'below','above'} if family == 'exact_numeric' else {'required_role_0','required_role_1'}
        if values != expected:raise ValueError('State-derived structural values are incomplete')
    structural_limits = [dict(split=split,family=family,condition=condition,axis=axis,
        missing=[key for key in ('choice','noul/no','noul/yes') if not counts[key]])
        for (split,family,condition,axis),counts in structures.items() if split in ('calibration','validation')]
    return {'schema':schema,'oracle_checked':len(records),'complete_groups':len(groups),
        'condition_kind_truth_coverage':{'/'.join(key):dict(value) for key,value in sorted(coverage.items())},
        'choice_target_positions':{'/'.join(map(str,key)):dict(value) for key,value in sorted(positions.items())},
        'state_derived_structural_coverage':{'/'.join(key):dict(value) for key,value in sorted(structures.items())},
        'structural_coverage_limits':structural_limits,
        'limits':['Calibration/Validation have only four groups per family; each binary structure has Choice and Noul but not both Noul truth values.',
                  'Scope field, affected role and question kind are not fully crossed; controlled OOD is not a natural-request or official benchmark result.']}


def audit_directory(directory):
    directory = Path(directory);manifest_path = directory/'manifest.json'
    manifest_sha256 = file_sha256(manifest_path);manifest = json.loads(manifest_path.read_text())
    expected_files = {split+'.jsonl' for split in SPLITS}
    if set(manifest['files_sha256']) != expected_files:raise ValueError('All five exact split files are required')
    records=[];files_sha256={}
    for split in SPLITS:
        path = directory/(split+'.jsonl');digest = file_sha256(path)
        if digest != manifest['files_sha256'][path.name]:raise ValueError('Candidate split hash differs: '+split)
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if any(row['split'] != split for row in values):raise ValueError('Row appears in the wrong split file')
        records.extend(values);files_sha256[path.name]=digest
    result = audit_rows(records)
    if manifest['summary'] != result['schema']:raise ValueError('Manifest schema summary differs from replayed records')
    if file_sha256(manifest_path) != manifest_sha256 or any(file_sha256(directory/name) != digest for name,digest in files_sha256.items()):
        raise ValueError('Candidate bytes changed during independent audit')
    return {'status':'validated_no_model_run','manifest_sha256':manifest_sha256,'files_sha256':files_sha256,
            'validator_sha256':file_sha256(__file__),**result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():raise FileExistsError('Refusing to overwrite an independent audit receipt')
    result = audit_directory(args.dataset)
    with output.open('x') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('status','oracle_checked','complete_groups')}))


if __name__ == '__main__':main()
