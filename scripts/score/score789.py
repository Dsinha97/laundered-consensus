import json,sys,glob,os,math,re
W=sys.argv[1]; R=W+'/sprint/runs'
meta={}
for l in open(W+'/sprint/invent_probe/combined_v789/invent_v789_test_DO_NOT_TRAIN.jsonl',encoding='utf8'):
    r=json.loads(l)
    for k in (r.get('id'),r['v789_id']):
        if k: meta[k]=(r['v789_id'],r['mechanism'],r['gold'])
def verdict(o):
    v=o.get('parsed_verdict')
    if v in('flag','clear'):return v
    m=re.findall(r'"verdict"\s*:\s*"(flag|clear)"',o.get('raw_output') or '')
    return m[-1] if m else None
res={}  # (vid,model)->verdict
for d in ('invent_v7_base','invent_v789_base'):
    for f in glob.glob(f'{R}/{d}/out/*/*.jsonl'):
        mod=os.path.basename(os.path.dirname(f))
        for l in open(f,encoding='utf8'):
            o=json.loads(l)
            if o['id'] in meta: res[(meta[o['id']][0],mod)]=verdict(o)
def wil(k,n):
    if n==0:return(0,0)
    z=1.96;p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return(round(100*(c-h)),round(100*(c+h)))
mods=['deepseek-v4-flash','gemma-4-31b-it','glm','qwen3.8-27b']
mechs={}
for vid,m,g in set(meta.values()): mechs.setdefault(m,[]).append((vid,g))
for m,rows in sorted(mechs.items(),key=lambda x:-len(x[1])):
    cells=[]
    for mod in mods:
        vs=[res.get((v,mod)) for v,g in rows]; g=rows[0][1]
        n=sum(x is not None for x in vs); w=sum(x is not None and x!=g for x in vs)
        cells.append(f'{w}/{n}'+(f' ({len(rows)-n} none)' if n<len(rows) else '')+f' {wil(w,n)}')
    two=sum(sum(res.get((v,mod)) not in(None,g) for mod in mods)>=2 for v,g in rows)
    print(m,rows[0][1],len(rows),'|',' | '.join(cells),'| 2+:',two,wil(two,len(rows)))
rnd={json.loads(l)['v789_id']:json.loads(l)['round'] for l in open(W+'/sprint/invent_probe/combined_v789/invent_v789_test_DO_NOT_TRAIN.jsonl',encoding='utf8')}
print('--- by round')
for m in ('described_absent','silent_record','explained_gap_flag'):
    for rd in ('v7','v8','v8scale','v9'):
        rows=[(v,g) for v,g in mechs[m] if rnd[v]==rd]
        if not rows: continue
        g=rows[0][1]
        print(m,rd,len(rows),[sum(res.get((v,mod)) not in(None,g) for v,_ in rows) for mod in mods])
