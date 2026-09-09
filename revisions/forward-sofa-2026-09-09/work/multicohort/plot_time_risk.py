"""Figure addressing removal of SOFA-derived predictors and time-specific eligibility."""
from pathlib import Path
import shutil
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('work/multicohort/results');F=Path('work/reviewer_revision/figures');F.mkdir(exist_ok=True)
m=pd.read_csv(R/'time_risk_metrics.csv');m=m.loc[m.target.eq('support')];flow=pd.read_csv(R/'time_risk_flow.csv');lead=pd.read_csv(R/'time_risk_lead_intervals.csv');tests=pd.read_csv(R/'time_risk_slope_tests.csv').set_index('hour')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7.5,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
# Retain the previous operational support-plus-SOFA analysis as a sensitivity figure.
for ext in ['png','pdf','svg']:
 target=F/('Supplementary_Figure_4_support_SOFA_sensitivity.'+ext)
 if not target.exists():shutil.copy2(F/('Figure_2_deterioration_warning.'+ext),target)
fig=plt.figure(figsize=(7.2,7.15));gs=fig.add_gridspec(3,2,height_ratios=[1.9,2.1,1.6],hspace=.42,wspace=.28)
ax=fig.add_subplot(gs[0,0]);x=np.arange(7)
ax.bar(x,flow.support_risk_n,color='#A3BDCE',width=.7)
for j,r in enumerate(flow.itertuples()):ax.text(j,r.support_risk_n+70,f'{r.support_risk_n:,}',ha='center',fontsize=7)
ax.set(xticks=x,xticklabels=flow.hour.astype(str),ylabel='Eligible event-free patients, n',xlabel='Prediction time after ICU admission (h)',ylim=(0,3800));ax.set_title('a  Risk sets reassessed at each time',loc='left',fontsize=9,fontweight='bold',pad=10)
ax=fig.add_subplot(gs[0,1]);colors={'history':'#326E91','sofa_level':'#B58A52','sofa_slope':'#714E70'}
for k,label in [('history','No SOFA: vital history'),('sofa_level','Vital history + latest SOFA'),('sofa_slope','History + SOFA level/slope')]:
 d=m.loc[m.model.eq(k)].set_index('hour').loc[flow.hour];ax.plot(x,d.auroc,marker='o',markersize=3,color=colors[k],lw=1.1,label=label)
ax.axhline(.5,color='#999999',ls=':',lw=.7);ax.set(xticks=x,xticklabels=flow.hour.astype(str),ylim=(.45,.73),yticks=[.5,.6,.7],ylabel='AUROC',xlabel='Prediction time after ICU admission (h)');ax.set_title('b  New support within the next 12 h',loc='left',fontsize=9,fontweight='bold',pad=10);ax.legend(loc='upper right',fontsize=6.7,frameon=False)

def table(ax,headers,rows,widths,fs):
 ax.axis('off');tab=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,0,1,1]);tab.auto_set_font_size(False);tab.set_fontsize(fs)
 for (i,j),c in tab.get_celld().items():
  c.set_edgecolor('white');c.set_linewidth(.5);c.PAD=.025;c.set_facecolor('#E7EDF1' if i==0 else ('#F4F6F8' if i%2 else 'white'))
  if i==0:c.set_text_props(weight='bold')
 return tab
rows=[]
for t in flow.hour:
 ds=m.loc[m.hour.eq(t)].set_index('model');r=ds.loc['history'];cells=[]
 for k in ['history','sofa_level','sofa_slope']:
  z=ds.loc[k];cells.append(f'{z.auroc:.3f}\n({z.auroc_lo:.3f}–{z.auroc_hi:.3f})')
 pv=tests.loc[t,'p_holm'];rows.append([str(t),f'{int(r.events)}/{int(r.n):,}',*cells,'<0.001' if pv<.001 else f'{pv:.3f}'])
ax=fig.add_subplot(gs[1,:]);table(ax,['Time\n(h)','Events / n','No SOFA\nAUC (95% CI)','+ latest SOFA\nAUC (95% CI)','+ SOFA slope\nAUC (95% CI)','Slope addition\nHolm P'],rows,[.07,.13,.2,.2,.2,.2],7)
rows=[]
for t in flow.hour:
 r=m.loc[m.hour.eq(t)&m.model.eq('history')].iloc[0];l=lead.loc[lead.hour.eq(t)&lead.model.eq('history')].iloc[0]
 rows.append([t,f'{int(r.tp)}/{int(r.events)}',f'{int(r.fp):,}',f'{100*r.sensitivity:.1f}',f'{100*r.ppv:.1f}',f'{l.median_h:.2f} ({l.q1_h:.2f}–{l.q3_h:.2f})'])
ax=fig.add_subplot(gs[2,:]);table(ax,['Time (h)','Detected / events','False alerts, n','Sensitivity %','PPV %','Alert interval, h\nmedian (IQR)'],rows,[.08,.19,.16,.15,.12,.3],7)
fig.text(.5,-.008,'Bottom table: vital-history model without SOFA or trajectory posteriors. Intervals concern true-positive alerts only.\nP compares adding SOFA slope to latest SOFA + vital history; Holm across 7 times. Conditional, pointwise CIs.\nRisk sets and event prevalence differ across times. No SOFA enters the event definition; discharge/death end observation.',ha='center',va='top',fontsize=6.85,linespacing=1.35)
for ext in ['png','pdf','svg']:fig.savefig(F/('Figure_2_deterioration_warning.'+ext),dpi=450,bbox_inches='tight',pad_inches=.08)
plt.close(fig)
