from pathlib import Path
import shutil
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path('work/future_trajectory');R=W/'results';F=W/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.7,'pdf.fonttype':42,'svg.fonttype':'none'})
cur=pd.read_csv(R/'outer_curves.csv');cl=pd.read_csv(R/'outer_classes.csv');m=pd.read_csv(R/'all_metrics.csv');mac=pd.read_csv(R/'all_multiclass.csv');tests=pd.read_csv(R/'all_tests.csv');flow=pd.read_csv(R/'all_flow.csv')
colors=['#326E91','#949274','#843D50']
fig=plt.figure(figsize=(7.2,6.9));gs=fig.add_gridspec(3,2,height_ratios=[1.25,2,1.5],hspace=.6,wspace=.3)
ax=fig.add_subplot(gs[0,:]);ax.set_title('a  Distinct prediction origins and future labels',loc='left',fontweight='bold',fontsize=10)
for y,t in [(1,12),(0,24)]:
 ax.plot([0,t],[y,y],lw=10,color='#A7C7CF',solid_capstyle='butt');ax.plot([t,60],[y,y],lw=10,color='#D9C4A6',solid_capstyle='butt');ax.axvline(t,ymin=0,ymax=1,color='#777777',ls=':',lw=.7)
 ax.text(t,y+.2,f'Predict at {t} h',ha='center',fontsize=8);ax.text(t/2,y-.26,'Prefix only',ha='center',fontsize=7.5);ax.text((t+60)/2,y-.26,f'Future SOFA >{t} to 60 h',ha='center',fontsize=7.5)
 ax.scatter(list(range(6,t+1,6)),[y]*int(t/6),color='#326E91',s=14,zorder=4)
ax.set(xlim=(0,61),ylim=(-.5,1.6),yticks=[],xticks=[0,6,12,24,36,48,60],xlabel='Hours from ICU admission')
for i,t in enumerate([12,24]):
 ax=fig.add_subplot(gs[1,i]);d=cur.loc[cur['mode'].eq('main')&cur.origin_h.eq(t)]
 for g,color in enumerate(colors,1):
  for fold,x in d.loc[d.target_class.eq(g)].groupby('outer_fold'):ax.plot(x.hour,x.fitted_sofa,color=color,alpha=.25,lw=.7)
  av=d.loc[d.target_class.eq(g)].groupby('hour').fitted_sofa.mean();ax.plot(av.index,av,color=color,lw=1.7,label=f'Class {g}')
 ax.set_title(f'{chr(98+i)}  Future classes after {t} h',loc='left',fontsize=10,fontweight='bold');ax.set(xlabel='Future score time (h)',ylabel='Fitted SOFA',ylim=(0,12),xticks=[18,24,36,48,60] if t==12 else [30,36,48,60]);ax.legend(frameon=False,fontsize=7,loc='lower left',ncol=3)
rows=[]
for t in [12,24]:
 for g in [1,2,3]:
  a=cl.loc[cl['mode'].eq('main')&cl.origin_h.eq(t)&cl.target_class.eq(g)];rows.append([f'{t} h',str(g),str(int(a.class_n.sum())),f'{a.first_future_sofa.mean():.2f} → {a.last_future_sofa.mean():.2f}',f'{a.net_change.min():+.3f} to {a.net_change.max():+.3f}',f'{int(a.rising.sum())}/5'])
ax=fig.add_subplot(gs[2,:]);ax.axis('off');tb=ax.table(cellText=rows,colLabels=['Origin','Class','OOF class n','First → last SOFA*','Net-change range*','Rising fits'],colWidths=[.10,.08,.13,.24,.28,.17],bbox=[0,0,1,1],cellLoc='center');tb.auto_set_font_size(False);tb.set_fontsize(7.2)
for (i,j),c in tb.get_celld().items():c.set_edgecolor('white');c.set_facecolor('#E7EDF1' if i==0 else ('#F3F5F6' if i%2 else 'white'));c.set_text_props(weight='bold' if i==0 else 'normal')
fig.text(.5,.06,'*Mean endpoints and range across the five outer training fits; thin lines show each fit.\nRising means fitted last minus first future SOFA >0; class 3 is highest severity, not automatically deterioration.\nDescriptive model curves: no curve-separation P test or AUC. Conventional SOFA uses rolling 24-hour source windows.',ha='center',va='top',fontsize=7)
for ext in ['png','pdf','svg']:fig.savefig(F/('Figure_1_forward_trajectories.'+ext),dpi=450,bbox_inches='tight',pad_inches=.08)
plt.close(fig)

fig=plt.figure(figsize=(7.2,7.4));gs=fig.add_gridspec(3,2,height_ratios=[2.1,2.3,1.6],hspace=.62,wspace=.32)
names=['history','latest_sofa','sofa_slope','prefix_added','prefix_only'];labels=['Vitals history','+ latest SOFA','+ SOFA slope','+ prefix posterior','Posterior only'];palette=['#777777','#326E91','#B58A52','#843D50','#568E83']
for i,t in enumerate([12,24]):
 ax=fig.add_subplot(gs[0,i]);d=mac.loc[mac['mode'].eq('main')&mac.origin_h.eq(t)].set_index('model').loc[names]
 for j,r in enumerate(d.itertuples()):
  ax.errorbar(r.macro_ovr_auroc,j,xerr=[[r.macro_ovr_auroc-r.macro_lo],[r.macro_hi-r.macro_ovr_auroc]],fmt='o',color=palette[j],capsize=2,markersize=4);ax.text(.97,j,f'{r.macro_ovr_auroc:.3f}',ha='right',va='center',fontsize=7)
 ax.set(yticks=range(5),yticklabels=labels if i==0 else ['']*5,xlim=(.5,.98),xticks=[.5,.7,.9],xlabel='Macro one-versus-rest AUROC');ax.invert_yaxis();ax.set_title(f'{chr(97+i)}  Predict at {t} h; n={int(d.n.iloc[0]):,}',fontsize=9,fontweight='bold',loc='left')
def draw_table(ax,heads,rows,widths,fs=7.1):
 ax.axis('off');tb=ax.table(cellText=rows,colLabels=heads,colWidths=widths,bbox=[0,0,1,1],cellLoc='center');tb.auto_set_font_size(False);tb.set_fontsize(fs)
 for (i,j),c in tb.get_celld().items():c.set_edgecolor('white');c.PAD=.03;c.set_facecolor('#E7EDF1' if i==0 else ('#F3F5F6' if i%2 else 'white'));c.set_text_props(weight='bold' if i==0 else 'normal')
rows=[]
for mode in ['main','separated']:
 for t in [12,24]:
  d=m.loc[m['mode'].eq(mode)&m.origin_h.eq(t)&m.target_class.eq(3)].set_index('model');q=tests.loc[tests['mode'].eq(mode)&tests.origin_h.eq(t)&tests.target_class.eq(3)&tests.addition.eq('prefix_added')].iloc[0]
  rows.append([f'{t} h\n'+('Main' if mode=='main' else 'Later target'),f'{int(d.loc["history","positive_n"])}/{int(d.loc["history","n"]):,}',*[f'{d.loc[k,"auroc"]:.3f}\n({d.loc[k,"auroc_lo"]:.3f}–{d.loc[k,"auroc_hi"]:.3f})' for k in ['history','latest_sofa','prefix_added']],f'{q.delta_auroc:+.5f}\nP={q.p_holm:.3f}'])
draw_table(fig.add_subplot(gs[1,:]),['Origin /\ntarget','High class\n/ evaluable','Vitals history\nAUC (95% CI)','+ latest SOFA\nAUC (95% CI)','+ prefix posterior\nAUC (95% CI)','Posterior gain\nHolm P'],rows,[.12,.14,.18,.18,.2,.18])
rows=[]
for mode in ['main','separated']:
 for t in [12,24]:
  r=flow.loc[flow['mode'].eq(mode)&flow.origin_h.eq(t)].iloc[0]
  rows.append([f'{t} h / '+('Main' if mode=='main' else 'Later'),int(r.recognized_present_n),int(r.prefix_eligible_n),int(r.future_evaluable_n),int(r.future_unassessable_n),f'{int(r.unassessable_death_by60)} / {int(r.unassessable_other_exit_by60)} / {int(r.unassessable_recording_gap)}'])
draw_table(fig.add_subplot(gs[2,:]),['Origin / target','Sepsis known\nand present','Prefix\neligible','Future\nevaluable','Unassessable','Unassessable: death /\nother exit / recording gap'],rows,[.15,.15,.12,.13,.15,.30],7.1)
fig.text(.5,.065,'Main future grids: >12 to 60 h and >24 to 60 h. Later targets start at 36/48 h, removing shared rolling score windows.\nHigh-class AUC is severity-pattern discrimination; a stable rising class was not recovered in the main analyses.\nConditional 95% CIs: 1,000 resamples. Posterior gain: 10,000 paired resamples; Holm across 12 tests per target family.\nPatients with unassessable future trajectories receive predictions but are excluded from accuracy estimates.',ha='center',va='top',fontsize=7,linespacing=1.25)
for ext in ['png','pdf','svg']:fig.savefig(F/('Figure_2_forward_prediction.'+ext),dpi=450,bbox_inches='tight',pad_inches=.08)
plt.close(fig)
old=Path('work/reviewer_revision/figures')
for name in ['Supplementary_Figure_1_trajectory_classification','Supplementary_Figure_2_mortality_ROC','Supplementary_Figure_3_mortality_calibration']:
 for ext in ['png','pdf','svg']:shutil.copy2(old/(name+'.'+ext),F/(name+'.'+ext))
for ext in ['png','pdf','svg']:shutil.copy2(old/('Figure_2_deterioration_warning.'+ext),F/('Supplementary_Figure_4_support_events.'+ext))
print('Two primary forward figures and four secondary figures prepared.')
