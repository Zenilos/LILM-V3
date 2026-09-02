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

for tag, path in [('fp_v2','checkpoints/fp_v2/best.npz'),
                  ('fp_sample','checkpoints/fp_sample/best.npz'),
                  ('fp_sample_last','checkpoints/fp_sample/last-14000.npz')]:
    if tag=='fp_sample_last':
        import os
        if not os.path.exists(path): print("skip", path); continue
    model = load_model(path); vocab=Vocab(); fsm=FSM(vocab)
    bt = build_base_tok([t for t,_ in rows])

    # atomic rows only: exactly one gold action, and not UNAVAILABLE
    atomic = [(t,g[0]) for t,g in rows if len(g)==1 and g[0].intent!='UNAVAILABLE']
    print(f"\n===== {tag}  atomic n={len(atomic)} =====")
    # overall intent accuracy on atomic
    intent_ok=0; intent_tot=0
    intent_c=collections.Counter(); intent_n=collections.Counter()
    exact=0  # intent correct (any slots)
    for text, g in atomic:
        words=text.split(); uid=bt.ids(words)
        pred = greedy_decode(model, words, uid, vocab, fsm, t=0.0)
        intent_n[g.intent]+=1
        if pred is not None and len(pred)==1 and pred[0].intent==g.intent:
            intent_c[g.intent]+=1; intent_ok+=1
        intent_tot+=1
    print(f"atomic intent-correct: {intent_ok}/{intent_tot} = {intent_ok/intent_tot:.1%}")
    print("per-intent (atomic rows):")
    for i in ['MOVE','CLEAN','PLAY','SHOW','HANDOVER','STOP','WAIT']:
        if intent_n[i]:
            print(f"   {i:<11} {intent_c[i]:>4}/{intent_n[i]:<4} {intent_c[i]/intent_n[i]:6.1%}")
