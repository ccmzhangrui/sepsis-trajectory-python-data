import runpy
runpy.run_path('work/multicohort/plot_numeric_figures.py',run_name='__main__')
raise SystemExit(0)

# Historical layout retained below for source history; not executed.
from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
W=Path('work/multicohort');M=W/'mimiciii';F=W/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
colors=['#426F92','#79A5AE','#A8A58C','#A36D65','#7C3544']
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(F/(name+'.'+ext),dpi=450,bbox_inches='tight')
 plt.close(fig)
curves=pd.read_csv(M/'results/trajectory_curves.csv');sel=pd.read_csv(M/'results/model_selection.csv');cl=pd.read_csv(M/'results/class_summary_k3.csv');mort=pd.read_csv(W/'results/mimiciii_mortality_by_trajectory.csv')
fig,axs=plt.subplots(2,2,figsize=(7.2,5.8),layout='constrained')
for ax,k in zip(axs[0],[3,5]):
 groups=pd.read_csv(M/f'results/class_summary_k{k}.csv')
 for j,row in groups.iterrows():
  d=curves.loc[curves.k.eq(k)&curves['class'].eq(row['class'])];col=colors[j] if k==5 else [colors[0],colors[2],colors[4]][j]
  ax.plot(d.hour,d.model_mean,lw=1.7,color=col,label=f'Class {j+1} (n={int(row.n):,})')
  ax.errorbar(d.hour,d.observed_mean,yerr=1.96*d.observed_sem,fmt='o',ms=2.3,elinewidth=.6,color=col)
 ax.set(xlabel='Hours from ICU admission',ylabel='Original SOFA score',ylim=(0,16),xticks=[6,12,24,36,48,60])
 ax.set_title('Three-class comparison' if k==3 else 'Five classes: lowest BIC tested',loc='left',fontsize=9,pad=9);ax.legend(fontsize=6.7,ncol=2)
ax=axs[1,0];ax.plot(sel.k,sel.bic-sel.bic.min(),'o-',color=colors[0]);ax.set(xlabel='Number of classes',ylabel='BIC above the minimum',xticks=[2,3,4,5])
ax=axs[1,1];d=mort.loc[mort.k.eq(3)];ax.errorbar(d['class'],100*d.mortality,yerr=np.array([100*(d.mortality-d.wilson_lo),100*(d.wilson_hi-d.mortality)]),fmt='o',ms=6,capsize=4,color=colors[4]);ax.set(xlabel='Full-window trajectory class',ylabel='Recorded hospital mortality (%)',xticks=[1,2,3],ylim=(0,50),xlim=(.5,3.5))
for i,ax in enumerate(axs.flat):ax.text(-.16,1.05,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=11)
save(fig,'Figure_1_trajectories')
# Separate cohorts and evaluation designs; no pooled cross-database curve.
pred={n:pd.read_pickle(W/f'private/{n}_mortality_oof.pkl') for n in ['mimiciii','eicu']};iv=np.load('work/reanalysis/results/local_predictions.npz');pred['mimiciv']=pd.DataFrame({'death':iv['y'],'demographics':iv['demographic'],'latest':iv['last_4bin'],'history':iv['history_4bin']})
models=['demographics','latest','history'];palette=['#929292','#BA8C53','#326E91'];labels=['Age and sex','Latest values','History summaries']
for kind in ['roc','calibration']:
 fig,axs=plt.subplots(1,3,figsize=(7.2,2.75),layout='constrained')
 for i,(ax,name) in enumerate(zip(axs,['mimiciii','mimiciv','eicu'])):
  p=pred[name]
  for model,color,label in zip(models,palette,labels):
   if kind=='roc':
    x,y,_=roc_curve(p.death,p[model]);ax.plot(x,y,color=color,lw=1.4,label=label)
   else:
    bins=np.array_split(np.argsort(p[model]),10);ax.plot([p[model].iloc[b].mean() for b in bins],[p.death.iloc[b].mean() for b in bins],'o-',color=color,lw=1.2,ms=3,label=label)
  ax.plot([0,1],[0,1],':',color='#aaaaaa',lw=.7);ax.set(xlim=(-.01,1.01),ylim=(-.01,1.01),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='False-positive rate' if kind=='roc' else 'Predicted probability',ylabel=('True-positive rate' if kind=='roc' else 'Observed mortality') if i==0 else '')
  title={'mimiciii':'MIMIC-III\n24 h | patient folds','mimiciv':'MIMIC-IV extract\nFour nominal slots | patient folds','eicu':'eICU\n24 h | hospital folds'}[name];ax.set_title(title,fontsize=8.3,loc='left',pad=10);ax.text(-.18,1.15,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=11)
 axs[0].legend(fontsize=6.4,loc='lower right' if kind=='roc' else 'upper left')
 save(fig,'Figure_2_mortality' if kind=='roc' else 'Supplementary_Figure_2_mortality_calibration')
p=pd.read_pickle(M/'private/trajectory_oof_predictions.pkl');names=['full_window_posterior_diagnostic','truncated_posterior_12h','truncated_posterior_24h','sofa_without_posterior','no_sofa_inputs'];labels=['Full-window diagnostic','Posterior window 0–12 h','Posterior window 0–24 h','SOFA, no posterior','No SOFA inputs'];pal=['#999999','#7C9BB3','#225A86','#BC8A4B','#377F74']
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
for name,label,col in zip(names,labels,pal):
 x,y,_=roc_curve(p.target,p[name]);axs[0].plot(x,y,color=col,label=label,lw=1.4)
 bins=np.array_split(np.argsort(p[name]),10);axs[1].plot([p[name].iloc[b].mean() for b in bins],[p.target.iloc[b].mean() for b in bins],'o-',ms=3,lw=1.2,color=col)
for i,ax in enumerate(axs):
 ax.plot([0,1],[0,1],':',color='#aaaaaa');ax.text(-.15,1.05,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=11)
axs[0].set(xlabel='False-positive rate',ylabel='True-positive rate');axs[0].legend(loc='lower right',fontsize=7)
axs[1].set(xlabel='Predicted probability',ylabel='Observed phenotype fraction');save(fig,'Supplementary_Figure_1_trajectory_classification')
print('Four figures saved as PNG, PDF and SVG.')
