import yaml

d = yaml.safe_load(open('/home/roger/a/bagla0/OpenSG-TW-claude/examples/data/iea_all_stations/shell51/1d_yaml/iea_s10_shell.yaml'))
print('reference field:', d.get('reference'))
for s in d['sections']:
    print(s['elementSet'], ':')
    tot = 0.0
    for p in s['layup']:
        print('   mat=%-24s t=%9.5f ang=%s' % (p[0], float(p[1]), p[2]))
        tot += float(p[1])
    print('   TOTAL %.4f m' % tot)
