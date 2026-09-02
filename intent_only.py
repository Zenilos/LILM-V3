import json, collections
from dsl import Action
from eval import load_model, greedy_decode
from train_student import build_base_tok
from serialize import Vocab, FSM

def load_rows(path):
    seen=set(); rows=[]
    for line in open(path):
        r=json.loads(line)
        if r['text'] in seen: continue
        seen.add(r['text'])
        rows.append((r['text'],[Action.from_dict(d) for d in r['actions']]))
    return rows

rows = load_rows('data/val.jsonl')
bt = build_base_tok([t for t,_ in rows])

for tag, path in [('fp_v2','checkpoints/fp_v2/best.npz'),
                  ('fp_sample','checkpoints/fp_sample/best.npz')]:
    model = load_model(path); vocab=Vocab(); fsm=FSM(vocab)
    total=0
    exact_seq=0
    len_ok=0
    len_ok_intent=0
    out_n=collections.Counter()
    # per gold action: did model hit intent (best-effort position match)
    intent_r = collections.defaultdict(lambda:[0,0])

    for text, gold in rows:
        words=text.split(); uid=bt.ids(words)
        pred = greedy_decode(model, words, uid, vocab, fsm, t=0.0)
        gold_ints=[a.intent for a in gold]
        total+=1
        if pred is None:
            out_n['none']+=1
            for gi in gold_ints: intent_r[gi][1]+=1
            continue
        pred_ints=[a.intent for a in pred]
        out_n['len_'+str(len(pred_ints))]+=1
        if pred_ints==gold_ints: exact_seq+=1
        if len(pred_ints)==len(gold_ints):
            len_ok+=1
            if pred_ints==gold_ints: len_ok_intent+=1
        for i,gi in enumerate(gold_ints):
            intent_r[gi][1]+=1
            if len(pred_ints)==len(gold_ints) and pred_ints[i]==gi:
                intent_r[gi][0]+=1

    print(f"\n===== {tag} =====")
    print(f"utterance intent-chain EXACT:  {exact_seq}/{total} = {exact_seq/total:.1%}")
    print(f"chain length correct:          {len_ok}/{total} = {len_ok/total:.1%}")
    print(f"length ok AND intent seq ok:   {len_ok_intent}/{total} = {len_ok_intent/total:.1%}")
    print(f"output sizes: {dict(out_n)}")
    print("per-intent (positional, only when length matches):")
    for i in ['MOVE','CLEAN','PLAY','SHOW','HANDOVER','STOP','WAIT','UNAVAILABLE']:
        if i in intent_r:
            c,t=intent_r[i]
            print(f"   {i:<11} {c:>4}/{t:<4} {c/t:6.1%}  (n={t})")
