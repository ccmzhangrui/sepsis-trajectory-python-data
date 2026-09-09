"""Numerical figures from observed cohort results and saved held-out predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

W=Path('work/multicohort'); M=W/'mimiciii'; F=W/'figures'; F.mkdir(exist_ok=True)
R=W/'results'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':8,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,
    'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
colors=['#426F92','#79A5AE','#A8A58C','#A36D65','#7C3544']
tests=pd.read_csv(R/'figure_paired_inference.csv')
association=pd.read_csv(R/'trajectory_mortality_association_tests.csv').set_index('k')
source_bins=[]

def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(F/(name+'.'+ext),dpi=450,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)

def ptext(value):
    return f'{value:.2e}' if value<.001 else f'{value:.3f}'

def letter(ax,i):
    ax.text(-.14,1.08,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=11)

def smalltable(ax,headers,rows,widths,font=7.6,colors_by_row=None):
    ax.axis('off')
    t=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='center',
               loc='center',bbox=[0,0,1,1])
    t.auto_set_font_size(False);t.set_fontsize(font)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('white');cell.set_linewidth(.5);cell.PAD=.035
        if i==0:
            cell.set_facecolor('#E7EDF1');cell.set_text_props(weight='bold')
        else:
            cell.set_facecolor('#F5F7F8' if i%2 else 'white')
            if j==0 and colors_by_row is not None:
                cell.set_text_props(color=colors_by_row[i-1],weight='bold')
    return t

def diag(ax,xlabel,ylabel):
    ax.plot([0,1],[0,1],':',color='#999999',lw=.8)
    ax.set(xlim=(-.015,1.015),ylim=(-.015,1.015),xticks=[0,.5,1],yticks=[0,.5,1],
           xlabel=xlabel,ylabel=ylabel)

def calibration(ax,p,y,cohort,model,color):
    bins=np.array_split(np.argsort(p,kind='stable'),10)
    xp=[float(np.mean(p[b])) for b in bins];yp=[float(np.mean(y[b])) for b in bins]
    ax.plot(xp,yp,'o-',color=color,lw=1.2,ms=3)
    for j,b in enumerate(bins):
        source_bins.append(dict(cohort=cohort,model=model,bin=j+1,n=len(b),
            events=int(y[b].sum()),predicted_mean=xp[j],observed_fraction=yp[j]))

# Figure 1: trajectory shapes, numerical fitted endpoints, BIC and descriptive mortality.
curves=pd.read_csv(M/'results/trajectory_curves.csv')
sel=pd.read_csv(M/'results/model_selection.csv')
mort=pd.read_csv(R/'mimiciii_mortality_by_trajectory.csv')
fig=plt.figure(figsize=(7.2,6.35))
gs=fig.add_gridspec(3,2,height_ratios=[2.2,.95,2.2],hspace=.52,wspace=.33)
for col,k in enumerate([3,5]):
    ax=fig.add_subplot(gs[0,col]);groups=pd.read_csv(M/f'results/class_summary_k{k}.csv')
    pal=colors if k==5 else [colors[0],colors[2],colors[4]];rows=[]
    for j,row in groups.iterrows():
        d=curves.loc[curves.k.eq(k)&curves['class'].eq(row['class'])]
        ax.plot(d.hour,d.model_mean,lw=1.6,color=pal[j])
        ax.errorbar(d.hour,d.observed_mean,yerr=1.96*d.observed_sem,fmt='o',ms=2.2,
                    elinewidth=.6,color=pal[j])
        rows.append([f'Class {j+1}',f'{int(row.n):,} ({100*row.n/3744:.1f}%)',
                    f'{d.model_mean.iloc[0]:.2f} → {d.model_mean.iloc[-1]:.2f}'])
    ax.set(xlabel='Hours from ICU admission',ylabel='Original SOFA score',ylim=(0,16),
           xticks=[6,12,24,36,48,60])
    ax.set_title('Three-class comparison' if k==3 else 'Five classes: lowest BIC tested',
                 loc='left',fontsize=9,pad=10);letter(ax,col)
    smalltable(fig.add_subplot(gs[1,col]),['Class','Patients, n (%)','SOFA 6 → 60 h'],
               rows,[.18,.39,.43],font=7.2,colors_by_row=pal)
ax=fig.add_subplot(gs[2,0]);ax.plot(sel.k,sel.bic,'o-',color=colors[0],ms=5)
for row in sel.itertuples():
    ax.annotate(f'{row.bic:,.1f}',(row.k,row.bic),xytext=(0,9),textcoords='offset points',
                ha='center',fontsize=7.6)
ax.set(xlabel='Number of classes',ylabel='BIC (lower is better)',xticks=[2,3,4,5],
       xlim=(1.65,5.35),ylim=(115000,139000))
ax.ticklabel_format(axis='y',style='plain',useOffset=False)
ax.set_title('Model selection',fontsize=9,loc='left',pad=10);letter(ax,2)
ax=fig.add_subplot(gs[2,1]);d=mort.loc[mort.k.eq(3)]
ax.errorbar(d['class'],100*d.mortality,
    yerr=np.array([100*(d.mortality-d.wilson_lo),100*(d.wilson_hi-d.mortality)]),
    fmt='o',ms=5,capsize=3,color=colors[4])
for _,row in d.iterrows():
    ax.text(row['class'],100*row.wilson_hi+2,
        f'{100*row.mortality:.1f}%\n({100*row.wilson_lo:.1f}–{100*row.wilson_hi:.1f})\n{int(row.deaths)}/{int(row.n):,}',
        ha='center',va='bottom',fontsize=7.2,linespacing=1.25)
at=association.loc[3]
ax.set(xlabel='Full-window trajectory class',ylabel='Hospital mortality (%)',
       xticks=[1,2,3],ylim=(0,67),xlim=(.4,3.6))
ax.set_title(f'Global χ²({int(at.df)}) = {at.chi_square:.2f}\nHolm P = {ptext(at.p_holm)}',
             fontsize=8.3,loc='left',pad=10);letter(ax,3)
fig.text(.5,-.025,'N = 3,744. Mortality labels: % (Wilson 95% CI), deaths/n.\n'
         'Trajectory curves and BIC are descriptive: no curve-separation P test; AUC not applicable.',
         ha='center',va='top',fontsize=7.5)
save(fig,'Figure_1_trajectories')

pred={n:pd.read_pickle(W/f'private/{n}_mortality_oof.pkl') for n in ['mimiciii','eicu']}
iv=np.load('work/reanalysis/results/local_predictions.npz')
pred['mimiciv']=pd.DataFrame({'death':iv['y'],'demographics':iv['demographic'],
    'latest':iv['last_4bin'],'history':iv['history_4bin']})
metrics={n:pd.read_csv(R/f'{n}_mortality_metrics.csv').set_index('model') for n in ['mimiciii','eicu']}
metrics['mimiciv']=pd.read_csv('work/reanalysis/results/metrics.csv').set_index('model').rename(
    index={'demographic':'demographics','last_4bin':'latest','history_4bin':'history'},
    columns={'calibration_intercept_joint':'calibration_intercept'})
models=['demographics','latest','history'];palette=['#8C8C8C','#B18242','#326E91']
labels=['Age and sex','Latest values','History summaries']

# Figures 2 and S2: separate numerical keys and explicit paired comparisons in each cohort.
for kind in ['roc','calibration']:
    fig=plt.figure(figsize=(7.2,5.7 if kind=='roc' else 6.2))
    gs=fig.add_gridspec(2,3,height_ratios=[2.1,2.55 if kind=='roc' else 3.05],hspace=.3,wspace=.3)
    for i,name in enumerate(['mimiciii','mimiciv','eicu']):
        ax=fig.add_subplot(gs[0,i]);p=pred[name];y=p.death.to_numpy();met=metrics[name]
        for model,color,label in zip(models,palette,labels):
            if kind=='roc':
                x,tpr,_=roc_curve(y,p[model]);ax.plot(x,tpr,color=color,lw=1.3)
            else:calibration(ax,p[model].to_numpy(),y,name,model,color)
        diag(ax,'False-positive rate' if kind=='roc' else 'Predicted probability',
             ('True-positive rate' if kind=='roc' else 'Observed mortality') if i==0 else '')
        title={'mimiciii':'MIMIC-III','mimiciv':'MIMIC-IV extract','eicu':'eICU'}[name]
        timing={'mimiciii':'24 h; patient folds','mimiciv':'Nominal slots; patient folds','eicu':'24 h; hospital folds'}[name]
        ax.set_title(f'{title}\nn = {len(p):,}; deaths = {int(y.sum()):,}\n{timing}',fontsize=8.1,loc='left',pad=12)
        letter(ax,i);key=fig.add_subplot(gs[1,i]);key.axis('off')
        for j,(model,color,label) in enumerate(zip(models,palette,labels)):
            r=met.loc[model];yy=.99-j*(.225 if kind=='roc' else .235)
            key.text(0,yy,label,transform=key.transAxes,color=color,weight='bold',fontsize=8,va='top')
            if kind=='roc':text=f'AUC {r.auroc:.3f} ({r.auroc_lo:.3f}–{r.auroc_hi:.3f})'
            else:text=(f'AUC {r.auroc:.3f}; Brier {r.brier:.4f}\n'
                       f'Intercept {r.calibration_intercept:.3f}; slope {r.calibration_slope:.3f}')
            key.text(0,yy-.065,text,transform=key.transAxes,fontsize=7.5,va='top',linespacing=1.5)
        metric='auroc' if kind=='roc' else 'brier'
        r=tests.loc[tests.cohort.eq(name)&tests.metric.eq(metric)].iloc[0]
        yy=.28 if kind=='roc' else .245
        key.text(0,yy,'History minus latest',transform=key.transAxes,fontsize=7.9,weight='bold',va='top')
        key.text(0,yy-.075,
            f'Δ{"AUC" if kind=="roc" else "Brier"} = {r.difference:+.4f}\n'
            f'95% CI {r.ci_low:+.4f} to {r.ci_high:+.4f}\nHolm P = {ptext(r.p_holm)}',
            transform=key.transAxes,fontsize=7.4,linespacing=1.55,va='top')
    fig.text(.5,.0,('AUC intervals: 95% CI. ' if kind=='roc' else 'P tests compare Brier error, not calibration intercept/slope. ')+
        'Two-sided paired bootstrap-normal tests.\nHolm adjustment across 9 mortality comparisons; '
        'conditional on fitted predictions. MIMIC-IV is retrospective sensitivity only.',
        ha='center',va='top',fontsize=7.15,linespacing=1.4)
    save(fig,'Supplementary_Figure_2_mortality_ROC' if kind=='roc' else 'Supplementary_Figure_3_mortality_calibration')

# Figure S1: full model key and three paired contrasts for phenotype classification.
p=pd.read_pickle(M/'private/trajectory_oof_predictions.pkl')
tm=pd.read_csv(M/'results/trajectory_classification_metrics.csv').set_index('variant')
names=['full_window_posterior_diagnostic','truncated_posterior_12h','truncated_posterior_24h',
       'sofa_without_posterior','no_sofa_inputs']
labels=['Full-window diagnostic','Posterior through 12 h','Posterior through 24 h',
        'SOFA, no posterior','No SOFA inputs']
pal=['#8C8C8C','#7C9BB3','#225A86','#BC8A4B','#377F74']
fig=plt.figure(figsize=(7.2,6.25));gs=fig.add_gridspec(3,2,height_ratios=[2.4,1.4,1.0],hspace=.38,wspace=.28)
axs=[fig.add_subplot(gs[0,i]) for i in [0,1]]
for name,label,col in zip(names,labels,pal):
    x,tpr,_=roc_curve(p.target,p[name]);axs[0].plot(x,tpr,color=col,lw=1.3)
    calibration(axs[1],p[name].to_numpy(),p.target.to_numpy(),'trajectory',name,col)
for i,ax in enumerate(axs):
    diag(ax,'False-positive rate' if i==0 else 'Predicted probability',
         'True-positive rate' if i==0 else 'Observed phenotype fraction');letter(ax,i)
    ax.set_title('Trajectory ROC' if i==0 else 'Trajectory calibration',loc='left',fontsize=9,pad=10)
rows=[]
for name,label in zip(names,labels):
    r=tm.loc[name];digits=6 if 'diagnostic' in name else 3
    rows.append([label,f'{r.auroc:.{digits}f} ({r.auroc_ci_low:.{digits}f}–{r.auroc_ci_high:.{digits}f})',
                 f'{r.brier:.4f} ({r.brier_ci_low:.4f}–{r.brier_ci_high:.4f})'])
smalltable(fig.add_subplot(gs[1,:]),['Model','AUC (95% CI)','Brier (95% CI)'],rows,
           [.36,.34,.30],font=7.65,colors_by_row=pal)
tr=tests.loc[tests.cohort.eq('trajectory')];rows=[]
for label,(_,r) in zip(['24 h posterior − SOFA, no posterior','12 h posterior − SOFA, no posterior',
                      '24 h posterior − no SOFA inputs'],tr.iterrows()):
    rows.append([label,f'{r.difference:+.5f} ({r.ci_low:+.5f} to {r.ci_high:+.5f})',ptext(r.p_holm)])
smalltable(fig.add_subplot(gs[2,:]),['Paired comparison','ΔAUC (95% CI)','Holm P'],rows,[.43,.40,.17],font=7.5)
fig.text(.5,-.005,'n = 3,744; fold-defined positives = 708. Prediction time: ICU hour 24.\n'
    'Full-window diagnostic intentionally uses future information. Two-sided bootstrap-normal P; Holm across 3 contrasts.\n'
    'Phenotype classification alone does not estimate clinical-event warning time.',
    ha='center',va='top',fontsize=7.2,linespacing=1.4)
save(fig,'Supplementary_Figure_1_trajectory_classification')
pd.DataFrame(source_bins).to_csv(R/'figure_calibration_bin_values.csv',index=False)
print('Four numerical figures saved as PNG, PDF and SVG; calibration-bin source values exported.')

# Main Figure 2 now centers on operational clinical deterioration and alert timing.
ep=pd.read_pickle(W/'private/deterioration_oof.pkl')
em=pd.read_csv(R/'deterioration_metrics.csv').set_index('model')
el=pd.read_csv(R/'deterioration_lead_time.csv').set_index('model')
et=pd.read_csv(R/'deterioration_paired_inference.csv')
names=['latest','history','sofa_history','prefix_trajectory']
labs=['Latest vital values','Vital history','Vital + SOFA history','History + trajectory']
pal=['#909090','#79A5AE','#BA8C53','#225A86']
fig=plt.figure(figsize=(7.2,6.8))
gs=fig.add_gridspec(3,2,height_ratios=[2.2,1.6,1.0],hspace=.35,wspace=.29)
ax=fig.add_subplot(gs[0,0]);lead_ax=fig.add_subplot(gs[0,1])
lead_curves=[]
for name,label,col in zip(names,labs,pal):
    x,tpr,_=roc_curve(ep.event,ep[name]);ax.plot(x,tpr,color=col,lw=1.4)
    hours=(ep.loc[ep.event.eq(1)&ep[name+'_alarm'],'event_hour']-24).to_numpy()
    x=np.r_[0,np.sort(hours),36]
    detection=np.array([(hours>=h).sum()/101 for h in x])
    lead_ax.step(x,100*detection,where='pre',color=col,lw=1.4)
    for h,v in zip(x,detection):lead_curves.append(dict(model=name,lead_hours=h,detected_fraction_all_events=v,all_events=101))
diag(ax,'False-positive rate','True-positive rate');letter(ax,0)
ax.set_title('Subsequent deterioration ROC',loc='left',fontsize=9,pad=10)
lead_ax.set(xlabel='Minimum interval before support onset (h)',ylabel='All events detected at least this early (%)',
    xlim=(0,36),ylim=(0,72),xticks=[0,6,12,24,36],yticks=[0,20,40,60])
lead_ax.set_title('Single alert at ICU hour 24',loc='left',fontsize=9,pad=10);letter(lead_ax,1)
lead_ax.axvline(6,color='#cccccc',ls=':',lw=.7);lead_ax.axvline(12,color='#cccccc',ls=':',lw=.7)
r=el.loc['prefix_trajectory']
lead_ax.text(.985,.94,f'Trajectory model\n≥6 h: 44/101 (43.6%)\n≥12 h: 30/101 (29.7%)',
    transform=lead_ax.transAxes,ha='right',va='top',fontsize=7.6,color=pal[-1],linespacing=1.4)
rows=[]
for name,label in zip(names,labs):
    r=em.loc[name];l=el.loc[name]
    rows.append([label,f'{r.auroc:.3f}\n({r.auroc_lo:.3f}–{r.auroc_hi:.3f})',
        f'{100*r.sensitivity:.1f}',f'{100*r.ppv:.1f}',f'{int(r.fp):,}',
        f'{l.median_hours:.2f}\n({l.q1_hours:.2f}–{l.q3_hours:.2f})'])
smalltable(fig.add_subplot(gs[1,:]),['Model','AUC (95% CI)','Sens. %','PPV %','FP, n','Interval h, median (IQR)'],
    rows,[.255,.21,.09,.085,.09,.27],font=7.3,colors_by_row=pal)
mapping=dict(zip(names,labs))
smalltable(fig.add_subplot(gs[2,:]),['Paired comparison','ΔAUC (95% CI)','Holm P'],
    [[mapping[r.model_a]+' − '+mapping[r.model_b],f'{r.difference:+.4f} ({r.ci_low:+.4f} to {r.ci_high:+.4f})',ptext(r.p_holm)] for r in et.itertuples()],
    [.47,.36,.17],font=7.3)
fig.text(.5,.003,'n = 3,697; events = 101 (2.73%): 51 ventilation, 50 vasopressor. Outcome window: (24, 60] h.\n'
    'Intervals summarize correctly flagged events only. Thresholds selected in inner folds; all-event denominator in panel b.\n'
    'Two-sided bootstrap-normal P; Holm across 3 AUC contrasts. Pointwise CIs; fitted-prediction uncertainty only.',
    ha='center',va='top',fontsize=7.1,linespacing=1.4)
save(fig,'Figure_2_deterioration_warning')
pd.DataFrame(lead_curves).to_csv(R/'deterioration_lead_curve_values.csv',index=False)
print('Main Figure 2 saved for operational deterioration and landmark alert intervals.')
