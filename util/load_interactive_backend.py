#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import chart_studio.plotly as py
import plotly.graph_objects as go
import ipywidgets as w
from IPython.display import display, clear_output
import seaborn as sns

from plotly.subplots import make_subplots
from enum import Enum
from itertools import repeat

print("Building the interface...")

def togglebtns_cb(change):
    if change.name!='index':
        return
    color_change(change)
    for update_fn in update_fns:
        update_fn(change)


## Generate legends
import matplotlib.pyplot as plt

sns.set()

_, ax = plt.subplots()
sns.scatterplot(
    x='pca_x', y='pca_y',
    alpha=0.7,
    hue='label', palette=sns.color_palette("Paired", 12),
    s=8,
    linewidth=0.2,
    data=df,
    ax=ax
)

handles, labels = ax.get_legend_handles_labels()

legend_class,ax = plt.subplots(figsize=(2, 1.25), dpi=300)

plt.rcParams.update({
    "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),
    "axes.facecolor":    (0.0, 0.0, 0.0, 0.0),
})

ax.legend(
    handles[1:], [("%s: " % lbl) + str(Label(int(lbl))) for lbl in labels[1:]],
    title="Class",
    loc='center',
    ncol=3,
    prop={'size': 6},
    frameon=False
)
plt.axis('off')

_, ax = plt.subplots()
sns.scatterplot(
    x='pca_x', y='pca_y',
    alpha=0.7,
    hue='session_uid', palette=sns.color_palette("Paired", 12),
    s=8,
    linewidth=0.2,
    data=df,
    ax=ax
)

handles, labels = ax.get_legend_handles_labels()
legend_session,ax = plt.subplots(figsize=(2, 1.25), dpi=300)
ax.legend(
    handles[1:], labels[1:],
    title="Session",
    loc='center',
    ncol=6,
    prop={'size': 6},
    frameon=False
)
plt.axis('off')


_, ax = plt.subplots()
sns.scatterplot(
    x='pca_x', y='pca_y',
    alpha=0.7,
    hue='iteration', palette=sns.color_palette("rocket", 8),
    s=8,
    linewidth=0.2,
    data=df,
    ax=ax
)

handles, labels = ax.get_legend_handles_labels()
legend_iteration,ax = plt.subplots(figsize=(2, 1.25), dpi=300)
ax.legend(
    handles[1:], labels[1:],
    title="Iteration",
    loc='center',
    ncol=8,
    prop={'size': 6},
    frameon=False
)
plt.axis('off')
plt.close('all')


def create_traces(hue, colors, mode):

    assert mode in ['pca', 'tsne'], "Invalid plot mode! Should be pca or tsne"

    if mode == 'pca':
        x = 'pca_x'
        y = 'pca_y'
    else:
        x = 'tsne_90_x'
        y = 'tsne_90_y'

    subjects_unique = list(df.subject.unique())
    sessions_unique = list(df.session_uid.unique())

    traces_raw = []
    traces_global = []
    traces_local = []

    for sub in df.subject.unique():
        for ses_ in df.query("subject==@sub").session.unique():
            for iter_ in df.query("subject==@sub and session==@ses_").iteration.unique():
                for cls in df.query("subject==@sub and session==@ses_ and iteration==@iter_").label.unique():
                    tmp_df = df.query("subject==@sub and session==@ses_ and iteration==@iter_ and label==@cls")

                    color_idx = None
                    if hue=='label':
                        color_idx=int(cls)
                        lgd_group="%s" % cls
                    elif hue=='session':
                        color_idx=sessions_unique.index(tmp_df.session_uid.unique()[0])
                        lgd_group="%s" % tmp_df.session_uid.unique()[0]
                    elif hue=='subject':
                        color_idx=subjects_unique.index(sub)
                        lgd_group="%s" % sub
                    elif hue=='iteration':
                        color_idx=int(iter_)-1
                        lgd_group="%s" % iter_

                    traces_raw.append(
                        {
                            'x': tmp_df[x],
                            'y': tmp_df[y],
                            'type': 'scattergl',
                            'mode': 'markers',
                            'name': '%s/%s/%s/%s' % (sub, ses_, iter_, cls),
                            'legendgroup': lgd_group,
                            'hovertemplate': '%{text} <extra></extra>',
                            'marker': {'color': colors[color_idx]},
                            'text': tmp_df.apply(
                                lambda x: '<b>Session UID</b>: %s<br><b>Iteration</b>: %s<br><b>Class</b>: %s' % (
                                x.session_uid, x.iteration, x.label
                            ), axis=1),
                        }
                    )

                    tmp_df = df_global.query("subject==@sub and session==@ses_ and iteration==@iter_ and label==@cls")
                    traces_global.append(
                        {
                            'x': tmp_df[x],
                            'y': tmp_df[y],
                            'type': 'scattergl',
                            'mode': 'markers',
                            'name': '%s/%s/%s/%s' % (sub, ses_, iter_, cls),
                            'legendgroup': lgd_group,
                            'hovertemplate': '%{text} <extra></extra>',
                            'marker': {'color': colors[color_idx]},
                            'text': tmp_df.apply(
                                lambda x: '<b>Session UID</b>: %s<br><b>Iteration</b>: %s<br><b>Class</b>: %s' % (
                                x.session_uid, x.iteration, x.label
                            ), axis=1),
                        }
                    )

                    tmp_df = df_local.query("subject==@sub and session==@ses_ and iteration==@iter_ and label==@cls")
                    traces_local.append(
                        {
                            'x': tmp_df[x],
                            'y': tmp_df[y],
                            'type': 'scattergl',
                            'mode': 'markers',
                            'name': '%s/%s/%s/%s' % (sub, ses_, iter_, cls),
                            'legendgroup': lgd_group,
                            'hovertemplate': '%{text} <extra></extra>',
                            'marker': {'color': colors[color_idx]},
                            'text': tmp_df.apply(
                                lambda x: '<b>Session UID</b>: %s<br><b>Iteration</b>: %s<br><b>Class</b>: %s' % (
                                x.session_uid, x.iteration, x.label
                            ), axis=1),
                        }
                    )

    return traces_raw, traces_global, traces_local

def generate_fig_widgets(tr1, tr2, tr3, mode):

    assert mode in ['pca', 'tsne'], "Invalid plot mode! Should be pca or tsne"

    margin=dict(l=6, r=6, t=40, b=5)

    g1_ = go.FigureWidget(
        data=tr1,
        layout=dict(
            showlegend=False,
            title="%s Raw" % mode.upper(),
            autosize=True,
            margin=margin
        )
    )

    g2_ = go.FigureWidget(
        data=tr2,
        layout=dict(
            showlegend=False,
            title="%s Global" % mode.upper(),
            autosize=True,
            margin=margin
        )
    )

    g3_ = go.FigureWidget(
        data=tr3,
        layout=dict(
            showlegend=False,
            title="%s Local" % mode.upper(),
            autosize=True,
            margin=margin
        )
    )

    return g1_, g2_, g3_


def toggle_trace(trace, crit):
    active_subs, active_its, active_classes, active_ses = crit
    subs_, session_, iters_, cls_ = trace.name.split("/")

    if subs_ in active_subs and iters_ in active_its and cls_ in active_classes and session_ in active_ses:
        trace.visible = True
    else:
        trace.visible = False

def change_subplot(change, fig):
    if change and change.name!='value':
        return

    active_subs = df.subject.unique().astype(str) if sub_dropdown.value == 'All' else [sub_dropdown.value]
    active_its = [x.description for x in it_buttons if x.value and not x.disabled]
    color = tglbtns.value
    change_color = isinstance(change.owner, w.ToggleButtons) if change else False
    active_ses = [x.description for x in ses_buttons if x.value and not x.disabled]
    active_classes = [x.description.split(':')[0] for x in class_buttons if x.value and not x.disabled]

    with fig.batch_update():
        list(map(toggle_trace, fig.data, repeat([active_subs, active_its, active_classes, active_ses])))


def change_raw(change):
    change_subplot(change, raw_box.children[0])

def change_global(change):
    change_subplot(change, global_box.children[0])

def change_local(change):
    change_subplot(change, local_box.children[0])

update_fns = [change_raw, change_global, change_local]

def apply_meta_fn(btn):
    change=None
    apply_btn.disabled=True
    apply_btn.icon='spinner'
    [update_fn(change) for update_fn in update_fns]
    apply_btn.icon='check'
    apply_btn.disabled=False

def some_btn_clicked(Change):
    apply_btn.icon=''

def toggle_automargins(change):
    if change.name!='value':
        return

    toggle_state = chkbx.value

    g_1 = raw_box.children[0]
    g_2 = global_box.children[0]
    g_3 = local_box.children[0]

    [g.update_xaxes(autorange=toggle_state) for g in [g_1, g_2, g_3]]
    [g.update_yaxes(autorange=toggle_state) for g in [g_1, g_2, g_3]]

def color_change(change, force=False):
    if not force and change.name!='index':
        return

    selection = tglbtns.options[tglbtns.index]

    if selection == 'Class':
        hue='label'
        colors=sns.color_palette("Paired", 12).as_hex()
        with out_legend:
            clear_output()
            display(legend_class)
    elif selection == 'Session':
        hue='session'
        colors=sns.color_palette("Paired", 12).as_hex()
        with out_legend:
            clear_output()
            display(legend_session)
    elif selection == 'Iteration':
        hue='iteration'
        colors=sns.color_palette("rocket", 8).as_hex()
        with out_legend:
            clear_output()
            display(legend_iteration)
    t1_, t2_, t3_ = create_traces(hue=hue, colors=colors, mode=mode_toggle.value)
    g_1, g_2, g_3 = generate_fig_widgets(t1_, t2_, t3_, mode=mode_toggle.value)

    raw_box.children = [g_1]
    global_box.children = [g_2]
    local_box.children = [g_3]

    apply_meta_fn(None)


it_buttons = []

for i in np.arange(1,9):
    it_buttons.append(
        w.ToggleButton(
            value=True,
            description='%s' % i,
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            continuous_update=False,
            layout=dict(flex='1 1 auto', width='auto'),
            width='auto'
        )
    )

ses_buttons = []
for i in np.arange(1,9):
    ses_buttons.append(
        w.ToggleButton(
            value=True,
            description='%s' % i,
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            continuous_update=False,
            layout=dict(flex='1 1 auto', width='auto'),
            width='auto'
        )
    )

class_buttons = []
for i in np.arange(12):
    class_buttons.append(
        w.ToggleButton(
            value=True,
            description='%s: %s' % (i, str(Label(i))),
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            continuous_update=False,
            layout=dict(flex='1 1 auto', width='auto'),
            width='auto'
        )
    )

[btn.observe(some_btn_clicked) for btn in it_buttons+ses_buttons+class_buttons]

class_container = w.GridspecLayout(6, 2)
for i in np.arange(6):
    class_container[i, 0] = class_buttons[i]
    class_container[i, 1] = class_buttons[i+6]

ses_container = w.HBox([w.Label("Toggle Sessions:")]+ses_buttons)
it_container = w.HBox([w.Label("Toggle Iterations:")]+it_buttons)

mode_toggle = w.ToggleButtons(
    options={'PCA': 'pca', 't-SNE': 'tsne'},
    description='',
    disabled=False,
    value='tsne'
)

tglbtns = w.ToggleButtons(
    options=['Class', 'Session', 'Iteration'],
    description='',
    disabled=False,
    value='Iteration'
)

ses_container = w.GridspecLayout(5, 2, grid_gap="2px", align_items='center', justify_items='center', width="95%")
ses_container[0,:] = w.Label("Toggle Sessions:", align_content='center')
for i in np.arange(4):
    ses_container[i+1, 0] = ses_buttons[i]
    ses_container[i+1, 1] = ses_buttons[i+4]

it_container = w.GridspecLayout(5,2, grid_gap="2px", align_items='center', justify_items='center', width="95%")
it_container[0,:] = w.Label("Toggle Iterations:", align_content='center')
for i in np.arange(4):
    it_container[i+1, 0] = it_buttons[i]
    it_container[i+1, 1] = it_buttons[i+4]


class_container = w.GridspecLayout(7, 2, grid_gap="2px", justify_content='center')
class_container[0,:] = w.Label("Toggle Classes:", align_content='center')
for i in np.arange(6):
    class_container[i+1, 0] = class_buttons[i]
    class_container[i+1, 1] = class_buttons[i+6]

nested_grid = w.GridspecLayout(1, 3, align_items='top', justify_items='center')
nested_grid[0, 0] = w.Dropdown(
            description='Subject(s): ',
            value='All',
            options=['All'] + df['subject'].unique().tolist(),
            layout=dict(width='max-content')
        )
nested_grid[0, 1] = ses_container
nested_grid[0, 2] = it_container


grid_2 = w.GridspecLayout(1, 2, align_items='top', justify_items='center', width="95%", grid_gap="20px")
grid_2[0, 0] = nested_grid
grid_2[:, 1] = class_container

sub_dropdown = nested_grid[0,0]
sub_dropdown.observe(some_btn_clicked)
chkbx = w.Checkbox(
    value=True,
    description='Auto adjust plot margins',
    disabled=False,
    indent=False
)

chkbx.observe(toggle_automargins)

t1_, t2_, t3_ = create_traces(hue='iteration', colors=sns.color_palette("rocket", 8).as_hex(), mode='tsne')
g1, g2, g3 = generate_fig_widgets(t1_, t2_, t3_, mode='tsne')

apply_btn = w.Button(
    description='Apply filter',
    disabled=False,
)

raw_box = w.HBox([g1], flex='1 1 auto', width='auto')
global_box = w.HBox([g2], flex='1 1 auto', width='auto')
local_box = w.HBox([g3], flex='1 1 auto', width='auto')

apply_btn.on_click(apply_meta_fn)
mode_toggle.observe(lambda x: color_change(None, True) if x.name == 'value' else None)

out_legend = w.Output()
with out_legend:
    clear_output()
    display(legend_iteration)

interface = w.VBox([
    grid_2,
    apply_btn,
    raw_box,
    global_box,
    local_box,
    out_legend,
    w.Label("Color by:"),
    w.HBox([tglbtns, chkbx, mode_toggle]),
])

tglbtns.observe(togglebtns_cb)

interface
