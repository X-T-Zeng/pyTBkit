"""Exact generator ranks and the difference between minimum and irredundant."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sympy as sp
from tbkit import FiniteGroup,cyclic_group,dihedral_group,matrix_group


def elementary(rank):
    values=np.arange(2**rank)
    return FiniteGroup(np.bitwise_xor(values[:,None],values),name=f'C2^{rank}')


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    sx=sp.Matrix([[0,1],[1,0]]);sy=sp.Matrix([[0,-sp.I],[sp.I,0]])
    quaternion=matrix_group([sp.I*sx,sp.I*sy],name='Q8').group
    groups=[cyclic_group(1),cyclic_group(6),dihedral_group(3),dihedral_group(4),quaternion,elementary(3),elementary(4)]
    expected=[0,1,2,2,2,3,4]
    results=[g.generator_search() for g in groups]
    assert [result.rank for result in results]==expected
    assert all(g.is_generating(result.generators) for g,result in zip(groups,results))
    six=groups[1]
    assert six.is_generating([2,3]) and not six.is_generating([2]) and not six.is_generating([3])
    assert six.minimum_generators()==(1,)
    case_data=[{'group':g.name,'order':g.order,'rank':r.rank,'generators':[str(x) for x in r.generators],
                'subgroup_states':r.subgroup_states,'extensions':r.extensions} for g,r in zip(groups,results)]
    style={'font.family':'DejaVu Sans','font.size':9.5,'mathtext.fontset':'stixsans',
           'axes.spines.top':False,'axes.spines.right':False,'axes.titlelocation':'left',
           'axes.titlesize':10.5,'axes.labelcolor':'#263442','text.color':'#263442',
           'xtick.color':'#4B5563','ytick.color':'#4B5563','legend.frameon':False,
           'legend.fontsize':8,'figure.constrained_layout.use':True,'pdf.fonttype':42,'svg.fonttype':'none'}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(9.2,3.8),gridspec_kw={'width_ratios':[1.05,1]})
        colors=['#0072B2','#0072B2','#756BB1','#756BB1','#756BB1','#009E73','#009E73']
        axes[0].bar(np.arange(len(groups)),expected,color=colors,width=.65)
        for i,value in enumerate(expected):axes[0].text(i,value+.08,str(value),ha='center',va='bottom',fontsize=9)
        labels=[r'$C_1$',r'$C_6$',r'$D_3$',r'$D_4$',r'$Q_8$',r'$C_2^3$',r'$C_2^4$']
        axes[0].set_xticks(np.arange(len(groups)),labels=[label+'\n'+str(g.order) for label,g in zip(labels,groups)])
        axes[0].set(title='(a)  Minimum generator number',ylabel=r'$d(G)$',xlabel=r'Group and order $|G|$',ylim=(0,4.7))
        axes[0].set_yticks(range(5));axes[0].tick_params(direction='out')
        axes[0].legend(handles=[Patch(color='#0072B2',label='cyclic'),Patch(color='#756BB1',label='nonabelian'),Patch(color='#009E73',label=r'elementary abelian')],loc='upper left')
        axes[1].axis('off');axes[1].set_title(r'(b)  Irredundant is not minimum: $C_6$')
        def node(x,y,text,color):
            return axes[1].text(x,y,text,transform=axes[1].transAxes,ha='center',va='center',fontsize=10,
                         bbox=dict(boxstyle='round,pad=.45',facecolor='#F5F8FB',edgecolor=color,linewidth=.8))
        def arrow(start,end):
            axes[1].annotate('',xy=end.get_position(),xytext=start.get_position(),xycoords='axes fraction',
                             arrowprops=dict(arrowstyle='->',color='#8995A3',lw=1,shrinkA=2,shrinkB=2,
                                             patchA=start.get_bbox_patch(),patchB=end.get_bbox_patch()))
        axes[1].text(.02,.9,'Irredundant: 2 generators',transform=axes[1].transAxes,color='#B04E08',fontsize=10)
        a=node(.1,.69,r'$\{0\}$','#D55E00');b=node(.43,.69,r'$\langle2\rangle$'+'\n'+r'$|H|=3$','#D55E00');c=node(.82,.69,r'$\langle2,3\rangle=C_6$','#D55E00')
        arrow(a,b);arrow(b,c)
        axes[1].text(.02,.37,'Global minimum: 1 generator',transform=axes[1].transAxes,color='#0072B2',fontsize=10)
        a=node(.1,.17,r'$\{0\}$','#0072B2');b=node(.7,.17,r'$\langle1\rangle=C_6$','#0072B2');arrow(a,b)
        for extension in ('png','pdf','svg'):fig.savefig(output/('group-generators.'+extension),dpi=300,bbox_inches='tight')
        plt.close(fig)
    summary={'cases':case_data,'cyclic6_irredundant_pair':[2,3],'cyclic6_minimum':[1],
             'definition':'minimum cardinality of a generating set; trivial group has rank zero',
             'search':'BFS over generated subgroup states, raises on budget exhaustion'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez(output/'group-generator-data.npz',orders=np.array([g.order for g in groups]),ranks=np.array(expected))
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='work/group-generators')
    run(parser.parse_args().output)
