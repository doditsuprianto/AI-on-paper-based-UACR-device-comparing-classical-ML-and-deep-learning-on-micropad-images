"""Sensitivity of the nested-CV SVM estimate to the position of the cleaning steps (outlier removal, ICC filter).

Run with the same software as the main analysis (scikit-learn 1.6.1; other versions change the inner folds and therefore the scores).
Inputs : ../outputs/0_Shared/cache/features_91614d0a03ef.pkl, ../outputs/A_ML/cache/nested_main_38b779d633.pkl (frozen fold plan), ../data/UACR_labels_deidentified.xlsx
Outputs: sensitivity_cleaning_folds.csv (frozen folds, scenarios A-E), sensitivity_cleaning_seeds.csv (five new fold sets, scenarios A, B, D, E)
Scenarios: A reported; B no outlier removal; C outliers detected in the training part only; D ICC filter in the training part only; E = C + D.
"""
import sys, json, time, warnings, numpy as np, pandas as pd
from pathlib import Path
HERE=Path(__file__).resolve().parent
warnings.filterwarnings('ignore')
from scipy.stats import rankdata
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score
SEED=42; CLASSES=['NORMAL','MICRO','MACRO']; C2I={c:i for i,c in enumerate(CLASSES)}
K_GRID=[5,10,20]; CORR=0.90; ICC_MIN=0.50
def _imp():
    return SimpleImputer(strategy='median', keep_empty_features=True)
class TopK(BaseEstimator, TransformerMixin):
    def __init__(self,k=10,corr_thr=0.90): self.k=k; self.corr_thr=corr_thr
    def fit(self,X,y):
        X=np.asarray(X,float); y=np.asarray(y)
        F,_=f_classif(X,y); F=np.nan_to_num(F)
        Xr,yr=rankdata(X,axis=0),rankdata(y); Xc,yc=Xr-Xr.mean(0),yr-yr.mean()
        rho=np.nan_to_num((Xc*yc[:,None]).sum(0)/(np.sqrt((Xc**2).sum(0)*(yc**2).sum())+1e-12))
        score=(rankdata(F)+rankdata(np.abs(rho)))/2.0; order=np.argsort(-score)
        Z=(X-X.mean(0))/(X.std(0)+1e-12); sel=[]
        for j in order:
            if len(sel)>=self.k: break
            if not sel: sel.append(j); continue
            if (np.abs(Z[:,sel].T@Z[:,j])/len(Z)).max()<self.corr_thr: sel.append(j)
        self.support_=np.array(sel,int); return self
    def transform(self,X): return np.asarray(X,float)[:,self.support_]
def make_pipe(clf,k=10):
    return Pipeline([('imp',_imp()),('rs',RobustScaler()),('sel',TopK(k=k,corr_thr=CORR)),('sc',StandardScaler()),('clf',clf)])
GRID={'clf__C':[1,10,100],'clf__gamma':['scale',0.05],'sel__k':K_GRID}

def icc1(d,cols,group='sample'):
    n_i=d.groupby(group).size(); keep=n_i[n_i>=2].index; dd=d[d[group].isin(keep)]
    X=dd[cols]; gmean=X.groupby(dd[group]).transform('mean'); N,a=len(dd),dd[group].nunique(); grand=X.mean()
    gmu=X.groupby(dd[group]).mean(); n_g=dd.groupby(group).size().reindex(gmu.index).values
    ssb=(n_g[:,None]*(gmu.values-grand.values)**2).sum(axis=0); ssw=((X.values-gmean.values)**2).sum(axis=0)
    msb,msw=ssb/(a-1),ssw/max(N-a,1); n0=(N-(n_g**2).sum()/N)/(a-1)
    return pd.Series((msb-msw)/(msb+(n0-1)*msw+1e-12),index=cols)
def icc_feats(d,feats):
    icc=icc1(d,feats); thr=ICC_MIN
    if (icc>=thr).sum()<20: thr=float(icc.quantile(0.5))
    return [f for f in feats if icc[f]>=thr]
def outlier_flags(d,feats):
    Xi=SimpleImputer(strategy='median').fit_transform(d[feats]); Xi=np.nan_to_num(StandardScaler().fit_transform(Xi))
    Zi=PCA(n_components=min(10,Xi.shape[1]),random_state=SEED).fit_transform(Xi)
    flag=np.zeros(len(d),bool)
    for c in CLASSES:
        ix=np.where((d.label==c).values)[0]
        if len(ix)>=20:
            iso=IsolationForest(contamination=0.03,random_state=SEED,n_estimators=300).fit(Zi[ix])
            flag[ix[iso.predict(Zi[ix])==-1]]=True
    tmp=d.assign(_o=flag)
    for s_,g in tmp.groupby('sample'):
        if g._o.all(): tmp.loc[g.index[0],'_o']=False
    return tmp['_o'].values

def load():
    f=pd.read_pickle(HERE/'..'/'outputs'/'0_Shared'/'cache'/'features_91614d0a03ef.pkl'); f=f[f.ok.astype(bool)]
    x=pd.read_excel(HERE/'..'/'data'/'UACR_labels_deidentified.xlsx'); x=x[pd.to_numeric(x['No Sampel'],errors='coerce').notna()].copy()
    x['sample']=pd.to_numeric(x['No Sampel']).astype(int)
    x['label']=x['Label Kelas'].astype(str).str.strip().str.upper().replace({'MIKRO':'MICRO','MAKRO':'MACRO'}); x['y']=x['label'].map(C2I)
    meta=['seq','sample','lux','ok','err']
    feat_all=[c for c in f.columns if c not in meta and not c.startswith('md5_') and not c.endswith(('__roi_method','__roi_area','__spec_frac'))]
    df=f.merge(x[['sample','label','y']],on='sample',how='inner').reset_index(drop=True)
    return df,feat_all

def clean_steps(df0,feat_all):
    feats=list(feat_all)
    d0=df0.copy(); d0[feats]=d0[feats].replace([np.inf,-np.inf],np.nan)
    d1=d0[d0[feats].notna().mean(axis=1)>0.5].copy()
    drop=set()
    for role in ('alb','cre'):
        g=d1.groupby(f'md5_{role}')['sample'].nunique(); cross=g[g>1].index
        if len(cross): drop|=set(d1[d1[f'md5_{role}'].isin(cross)].index)
    same=d1.drop(index=list(drop)).duplicated(['sample','md5_alb','md5_cre'],keep='first'); drop|=set(same[same].index)
    d2=d1.drop(index=list(drop)).copy()
    miss=d2[feats].isna().mean(); nun=d2[feats].nunique(dropna=True)
    bad=sorted(set(miss[miss>0.20].index)|set(nun[nun<=2].index)); feats=[f for f in feats if f not in bad]
    return d2,feats,len(drop)

def nested_fold(Xtr,ytr,gtr,Xte,rep):
    pipe=make_pipe(SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=SEED),k=K_GRID[0])
    spl=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED+rep).split(Xtr,ytr,gtr))
    gs=GridSearchCV(pipe,GRID,scoring='f1_macro',cv=spl,n_jobs=1,refit=True).fit(Xtr,ytr)
    return gs.predict_proba(Xte), gs.best_params_['sel__k']

# ---------------- analysis ----------------

from sklearn.metrics import balanced_accuracy_score, recall_score, accuracy_score
df0,feat_all=load(); d2,feats,_=clean_steps(df0,feat_all); d2=d2.reset_index(drop=True)
fl=outlier_flags(d2,feats); keep=np.where(~fl)[0]; d3=d2.iloc[keep].reset_index(drop=True)
F_glob=icc_feats(d3,feats); F_all=icc_feats(d2,feats)
print('global ICC features: cleaned',len(F_glob),'| all 494 tests',len(F_all))
N=pd.read_pickle(HERE/'..'/'outputs'/'A_ML'/'cache'/'nested_main_38b779d633.pkl')
ref={(r['rep'],r['fold']):r for r in N if r['model']=='SVM (RBF)'}
g3=d3['sample'].values
plan=[(rep,fold,set(g3[r['tei']])) for (rep,fold),r in sorted(ref.items())]
ref_mask_d2=np.zeros(len(d2),bool); ref_mask_d2[keep]=True
def job(variant,rep,fold,tsamp):
    oo,ii=VAR[variant]
    if oo=='global': base=d3
    else: base=d2
    te=base[base['sample'].isin(tsamp)]; tr=base[~base['sample'].isin(tsamp)]
    if oo=='train':
        f=outlier_flags(tr.reset_index(drop=True),feats); tr=tr.reset_index(drop=True)[~f]
    if ii=='global': cols=F_glob if oo!='none' else F_all
    else: cols=icc_feats(tr,feats)
    p,k=nested_fold(tr[cols].values,tr.y.values.astype(int),tr['sample'].values,te[cols].values,rep)
    yt=te.y.values.astype(int); yp=p.argmax(1)
    ref_te=te[te['seq'].astype(str)+'_'+te['sample'].astype(str)].index if False else None
    keyset=set(zip(d3['seq'],d3['sample']))
    m=np.array([(s,sa) in keyset for s,sa in zip(te['seq'],te['sample'])])
    out=dict(variant=variant,rep=rep,fold=fold,n_train=len(tr),n_test=len(te),n_feat=len(cols),k=int(k),
        f1_all=f1_score(yt,yp,average='macro'),bacc_all=balanced_accuracy_score(yt,yp),
        f1_ref=f1_score(yt[m],yp[m],average='macro'),n_test_ref=int(m.sum()),
        rec_micro_ref=float(recall_score(yt[m],yp[m],labels=[1],average=None,zero_division=0)[0]))
    return out

if __name__=='__main__':
    VAR={'A':('global','global'),'B':('none','global'),'C':('train','global'),'D':('global','train'),'E':('train','train')}
    jobs=[(v,r,f,s) for v in VAR for (r,f,s) in plan]
    t=time.time(); res=Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs); print('time',time.time()-t)
    R=pd.DataFrame(res); R.to_csv(HERE/'sensitivity_cleaning_folds.csv',index=False)
    S=R.groupby('variant').agg(f1_all_mean=('f1_all','mean'),f1_all_sd=('f1_all','std'),f1_ref_mean=('f1_ref','mean'),f1_ref_sd=('f1_ref','std'),
        bacc=('bacc_all','mean'),micro=('rec_micro_ref','mean'),nfeat=('n_feat','mean'),ntest=('n_test','sum')).round(4)
    print(S)
    A=R[R.variant=='A'].sort_values(['rep','fold']).f1_ref.values
    for v in 'BCDE':
        x=R[R.variant==v].sort_values(['rep','fold']).f1_ref.values; d=x-A
        print(v,'delta vs A (ref tests): mean %.4f sd %.4f; lower in %d of 10 folds; min %.3f max %.3f'%(d.mean(),d.std(ddof=1),(d<0).sum(),d.min(),d.max()))
    
    VAR={'A':('global','global'),'B':('none','global'),'D':('global','train'),'E':('train','train')}
    y3=d3.y.values.astype(int)
    plan2=[]
    for s in range(5):
        for fold,(tri,tei) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=2000+s).split(np.zeros((len(y3),1)),y3,g3)):
            plan2.append((s,fold,set(g3[tei])))
    jobs=[(v,s,f,ts) for v in VAR for (s,f,ts) in plan2]
    t=time.time(); res=Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs); print('time',time.time()-t)
    R=pd.DataFrame(res); R.to_csv(HERE/'sensitivity_cleaning_seeds.csv',index=False)
    S=R.groupby('variant').agg(f1_ref_mean=('f1_ref','mean'),f1_ref_sd=('f1_ref','std'),f1_all_mean=('f1_all','mean')).round(4); print(S)
    print(R.groupby(['variant','rep']).f1_ref.mean().unstack().round(3))
    A=R[R.variant=='A'].sort_values(['rep','fold']).f1_ref.values
    for v in 'BDE':
        x=R[R.variant==v].sort_values(['rep','fold']).f1_ref.values; d=x-A
        print(v,'delta vs A: mean %.4f sd %.4f lower in %d of 25'%(d.mean(),d.std(ddof=1),(d<0).sum()))
