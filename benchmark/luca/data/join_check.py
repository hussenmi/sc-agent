import gzip, h5py, numpy as np, re
geo_idx=set()
with gzip.open('GSE131907_cell_annotation.txt.gz','rt') as f:
    f.readline()
    for line in f:
        geo_idx.add(line.split('\t',1)[0])
path='/data1/peerd/adamsj5/spatial_celltypist/spatial_celltypist/data_spatial/core_nsclc_atlas.h5ad'
with h5py.File(path,'r') as f:
    idx=np.array([x.decode() for x in f['obs']['_index'][:]])
    g=f['obs']['study']; cats=[c.decode() for c in g['categories'][:]]
    atlas_bc=idx[g['codes'][:]==cats.index('Kim_Lee_2020')]
stripped=[re.sub(r'-\d+$','',b) for b in atlas_bc]
hit=sum(1 for s in stripped if s in geo_idx)
print(f'GEO={len(geo_idx)} atlas={len(atlas_bc)} join={hit} pct={100*hit/len(atlas_bc):.1f}')
