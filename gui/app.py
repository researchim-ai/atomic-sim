
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nicegui import ui
import plotly.graph_objects as go
import numpy as np
from reactor import ReactorSimulator1D

# === Localization ===
LANG = {
    'en': {
        'title': 'ATOMIC SIM',
        'status_norm': 'SYSTEM NORMAL',
        'status_crit': 'CRITICAL WARNING',
        'ctrl_room': 'CONTROL ROOM',
        'tab_main': 'Main',
        'tab_safety': 'Safety',
        'tab_auto': 'Auto',
        'lbl_rods': 'Control Rods Position',
        'lbl_boron': 'Boron Concentration (ppm)',
        'lbl_auto_sys': 'Automatic Control System',
        'sw_pid': 'Power PID Control',
        'sw_shim': 'Boron Shim Control',
        'lbl_target': 'Target Power',
        'lbl_emergency': 'Emergency Systems',
        'btn_scram': 'MANUAL SCRAM',
        'lbl_faults': 'Experimental Faults',
        'btn_trip': 'Trip Coolant Pump',
        'btn_eject': 'Eject Control Rod',
        'kpi_power': 'Thermal Power',
        'kpi_temp': 'Peak Fuel Temp',
        'kpi_void': 'Void Fraction',
        'chart_core': 'Reactor Core Status (Axial Profile)',
        'chart_trend': 'Trends',
        'lbl_logs': 'System Event Log',
        'axis_height': 'Core Height (Bottom -> Top)',
        'axis_power': 'Power Density (MW/m)',
        'axis_temp': 'Temperature (°C)',
        'msg_scram': 'SCRAM INITIATED!',
        'msg_pump': 'Main Coolant Pump TRIPPED!',
        'msg_rod': 'Rod Ejection Accident!',
        'msg_overheat': 'CRITICAL TEMPERATURE WARNING! CORE DAMAGE RISK!',
        'legend_power': 'Power',
        'legend_tf': 'Fuel Temp',
        'legend_tc': 'Coolant Temp',
        'lbl_rods_short': 'RODS'
    },
    'ru': {
        'title': 'АТОМНЫЙ СИМУЛЯТОР',
        'status_norm': 'НОРМАЛЬНАЯ РАБОТА',
        'status_crit': 'КРИТИЧЕСКАЯ ТРЕВОГА',
        'ctrl_room': 'ПУЛЬТ УПРАВЛЕНИЯ',
        'tab_main': 'Главная',
        'tab_safety': 'Защита',
        'tab_auto': 'Авто',
        'lbl_rods': 'Положение Стержней СУЗ',
        'lbl_boron': 'Концентрация Бора (ppm)',
        'lbl_auto_sys': 'Система Автоматического Управления',
        'sw_pid': 'PID Регулятор Мощности',
        'sw_shim': 'Борное Регулирование',
        'lbl_target': 'Целевая Мощность',
        'lbl_emergency': 'Системы Безопасности',
        'btn_scram': 'АЗ (АВАРИЙНАЯ ЗАЩИТА)',
        'lbl_faults': 'Экспериментальные Отказы',
        'btn_trip': 'Отключить ГЦН (Насос)',
        'btn_eject': 'Выброс Стержня',
        'kpi_power': 'Тепловая Мощность',
        'kpi_temp': 'Макс. Темп. Топлива',
        'kpi_void': 'Паросодержание',
        'chart_core': 'Состояние Активной Зоны (Профиль)',
        'chart_trend': 'Тренды',
        'lbl_logs': 'Журнал Событий',
        'axis_height': 'Высота Активной Зоны (Низ -> Верх)',
        'axis_power': 'Плотность Мощности (МВт/м)',
        'axis_temp': 'Температура (°C)',
        'msg_scram': 'АВАРИЙНАЯ ЗАЩИТА АКТИВИРОВАНА!',
        'msg_pump': 'ОТКАЗ ГЛАВНОГО ЦИРКУЛЯЦИОННОГО НАСОСА!',
        'msg_rod': 'ВЫБРОС СТЕРЖНЯ СУЗ!',
        'msg_overheat': 'КРИТИЧЕСКИЙ ПЕРЕГРЕВ! РИСК ПОВРЕЖДЕНИЯ!',
        'legend_power': 'Мощность',
        'legend_tf': 'Т. Топлива',
        'legend_tc': 'Т. Воды',
        'lbl_rods_short': 'СУЗ'
    }
}

# Current language state
current_lang = 'ru' # Default to Russian as requested

def T(key):
    """Get translated string"""
    return LANG[current_lang].get(key, key)

# === Global Simulation State ===
sim = ReactorSimulator1D(num_nodes=50, device='cpu')
sim.boron_concentration = 1000.0
sim.rod_position = 0.2
# Stabilize
for _ in range(100):
    sim.step(dt=0.01)

# History Buffers
max_history = 300
time_hist = []
power_hist = []
fuel_temp_hist = []
boron_hist = []

# Event Log List
system_logs = []

def add_log(msg, level='info'):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {'time': timestamp, 'msg': msg, 'level': level}
    system_logs.insert(0, entry) # Prepend
    if len(system_logs) > 50: system_logs.pop()

# === UI Components ===

@ui.page('/')
def main_page():
    # Styling
    ui.colors(primary='#3b82f6', secondary='#64748b', accent='#f59e0b', dark='#0f172a')
    ui.query('body').style('background-color: #0b0f19; color: #e2e8f0; font-family: "Inter", sans-serif;')
    
    # Refreshable UI container
    # Note: Header cannot be inside a container in NiceGUI. It must be top-level.
    # We will only refresh the content below header.
    
    # --- HEADER (Static, but content can be updated via bindings/set_text) ---
    with ui.header().classes('bg-gray-900 items-center justify-between elevation-4 q-py-sm'):
        with ui.row().classes('items-center gap-4'):
            ui.icon('science', size='2rem').classes('text-blue-400')
            title_label = ui.label(T('title')).classes('text-xl font-bold tracking-wider text-blue-100')
            ui.badge('v1.2.0', color='blue-800').classes('text-xs')
        
        # Language Switcher
        with ui.row().classes('items-center gap-2'):
            def set_lang(l):
                global current_lang
                current_lang = l
                # Refresh UI
                render_ui() 
                # Update static header texts manually
                title_label.set_text(T('title'))
            
            ui.button('RU', on_click=lambda: set_lang('ru')).props('flat dense').classes('text-white')
            ui.label('|').classes('text-gray-600')
            ui.button('EN', on_click=lambda: set_lang('en')).props('flat dense').classes('text-white')

        # Status & Time
        with ui.row().classes('items-center gap-6'):
            # NiceGUI newer versions require sanitize=False for raw HTML styles
            alarm_lamp = ui.html('<div class="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_10px_rgba(74,222,128,0.8)]"></div>', sanitize=False)
            status_label = ui.label(T('status_norm')).classes('text-green-400 font-mono font-bold')
            ui.label('TIME: 00:00:00').classes('font-mono text-gray-400').bind_text_from(sim, 'time', lambda t: f"T+ {t:.1f}s")

    # Define alarm_lamp outside render_ui to be accessible?
    # Actually, update_loop is defined INSIDE render_ui, so it should see alarm_lamp if it is defined in render_ui scope.
    # BUT, update_loop is called by timer.
    
    # Let's make UI elements accessible via a simple class or dictionary storage for the page instance.
    ui_refs = {}

    main_container = ui.element('div').classes('w-full h-full')

    def render_ui():
        main_container.clear()
        with main_container:
            # --- GRID LAYOUT ---
            with ui.grid(columns='350px 1fr 300px').classes('w-full h-[calc(100vh-60px)] gap-4 p-4'):
                
                # === LEFT COL: CONTROLS ===
                with ui.card().classes('h-full bg-gray-900/50 border border-gray-800 flex flex-col gap-4'):
                    ui.label(T('ctrl_room')).classes('text-lg font-bold text-blue-400 border-b border-gray-800 pb-2')
                    
                    with ui.tabs().classes('w-full text-blue-400') as tabs:
                        main_tab = ui.tab(T('tab_main'))
                        safety_tab = ui.tab(T('tab_safety'))
                        auto_tab = ui.tab(T('tab_auto'))

                    with ui.tab_panels(tabs, value=main_tab).classes('w-full bg-transparent flex-grow'):
                        
                        # MAIN
                        with ui.tab_panel(main_tab).classes('flex flex-col gap-6'):
                            with ui.column().classes('w-full'):
                                ui.label(T('lbl_rods')).classes('text-sm text-gray-400 font-bold')
                                slider_rods = ui.slider(min=0, max=100, value=sim.rod_position*100, step=0.1).props('label-always color=orange track-size=8px thumb-size=20px')
                                
                                def manual_rods(e):
                                    if not sim.control.auto_power:
                                        sim.rod_position = slider_rods.value / 100.0
                                slider_rods.on('change', manual_rods)
                                
                                # Visual Bar
                                ui.linear_progress(value=0.2).bind_value_from(sim, 'rod_position').props('color=orange track-color=gray-800 size=15px rounded')

                            with ui.column().classes('w-full'):
                                ui.label(T('lbl_boron')).classes('text-sm text-gray-400 font-bold')
                                
                                def set_boron(e):
                                    try: sim.boron_concentration = float(boron_input.value)
                                    except: pass
                                    
                                boron_input = ui.number(value=sim.boron_concentration, format='%.0f', on_change=set_boron).classes('w-full text-lg').props('standout outlined dense')
                                ui.slider(min=0, max=2000, value=1000).bind_value(boron_input).props('color=cyan')

                        # AUTO
                        with ui.tab_panel(auto_tab).classes('flex flex-col gap-4'):
                            ui.label(T('lbl_auto_sys')).classes('font-bold text-center mb-2')
                            
                            sw_p = ui.switch(T('sw_pid'), value=sim.control.auto_power).classes('text-green-400')
                            sw_b = ui.switch(T('sw_shim'), value=sim.control.auto_boron).classes('text-cyan-400')
                            
                            ui.separator().classes('bg-gray-800')
                            ui.label(T('lbl_target') + ' (MW)').classes('text-sm text-gray-400')
                            target_p = ui.number(value=sim.control.target_power, step=100).classes('w-full').props('outlined dense suffix="MW"')
                            
                            def update_auto_settings():
                                sim.control.auto_power = sw_p.value
                                sim.control.auto_boron = sw_b.value
                                sim.control.target_power = target_p.value
                                slider_rods.disable() if sw_p.value else slider_rods.enable()
                                boron_input.disable() if sw_b.value else boron_input.enable()

                            sw_p.on('update:model-value', update_auto_settings)
                            sw_b.on('update:model-value', update_auto_settings)
                            target_p.on('change', update_auto_settings)

                        # SAFETY
                        with ui.tab_panel(safety_tab).classes('flex flex-col gap-4'):
                            ui.label(T('lbl_emergency')).classes('text-red-500 font-bold')
                            
                            def scram():
                                sim.rod_position = 1.0
                                slider_rods.value = 100
                                sw_p.value = False
                                sim.control.auto_power = False
                                add_log(T('msg_scram'), 'critical')
                            
                            ui.button(T('btn_scram'), on_click=scram).classes('bg-red-600 w-full font-bold py-4 text-lg shadow-[0_0_15px_rgba(220,38,38,0.6)] hover:bg-red-700 transition-all')

                            ui.separator().classes('bg-gray-800 my-2')
                            ui.label(T('lbl_faults')).classes('text-orange-400 font-bold')
                            
                            def trip_pump():
                                sim.thermal.flow_factor = 0.1
                                add_log(T('msg_pump'), 'warning')
                            
                            ui.button(T('btn_trip'), on_click=trip_pump).props('outline color=orange').classes('w-full')
                            
                            def eject_rod():
                                sim.rod_position = 0.0
                                slider_rods.value = 0
                                add_log(T('msg_rod'), 'critical')
                                
                            ui.button(T('btn_eject'), on_click=eject_rod).props('outline color=red').classes('w-full')

                # === MIDDLE COL: CORE VIZ ===
                with ui.column().classes('h-full gap-4'):
                    # KPI ROW
                    with ui.row().classes('w-full h-[120px] gap-4'):
                        def kpi_card(title, unit, color):
                            with ui.card().classes(f'flex-1 h-full bg-gray-900/50 border-l-4 border-{color}-500 flex flex-col justify-center items-center'):
                                ui.label(title).classes('text-gray-400 text-xs uppercase tracking-widest text-center')
                                lbl = ui.label('0').classes(f'text-4xl font-mono font-bold text-{color}-400')
                                ui.label(unit).classes(f'text-{color}-600 text-sm font-bold')
                                return lbl

                        lbl_power = kpi_card(T('kpi_power'), 'MW', 'orange')
                        lbl_temp = kpi_card(T('kpi_temp'), '°C', 'red')
                        lbl_void = kpi_card(T('kpi_void'), '%', 'purple')

                    # CHART AREA with Side Bar for Rods
                    with ui.row().classes('w-full flex-grow bg-gray-900 border border-gray-800 p-0 overflow-hidden relative no-wrap'):
                        # Main Chart
                        core_plot = ui.plotly({
                            'layout': {
                                'title': {'text': T('chart_core'), 'font': {'color': '#94a3b8'}},
                                'paper_bgcolor': 'rgba(0,0,0,0)',
                                'plot_bgcolor': 'rgba(0,0,0,0)',
                                'font': {'color': '#cbd5e1'},
                                'margin': {'l': 50, 'r': 20, 't': 50, 'b': 40},
                                'xaxis': {'title': T('axis_height'), 'range': [50, 0], 'gridcolor': '#334155'}, 
                                'yaxis': {'title': T('axis_power'), 'gridcolor': '#334155'},
                                'yaxis2': {'title': T('axis_temp'), 'overlaying': 'y', 'side': 'right', 'gridcolor': '#334155', 'showgrid': False},
                                'showlegend': True,
                                'legend': {'x': 0.02, 'y': 0.98, 'bgcolor': 'rgba(15,23,42,0.8)'}
                            }
                        }).classes('flex-grow h-full')
                        
                        # Rods Visualization Bar (Side Strip)
                        with ui.column().classes('w-[60px] h-full border-l border-gray-700 items-center p-2 bg-gray-800/30'):
                            ui.label(T('lbl_rods_short')).classes('text-[10px] font-bold text-gray-400 mb-1')
                            with ui.element('div').classes('w-[20px] flex-grow bg-gray-700 rounded relative overflow-hidden'):
                                # The Rod (Top-down)
                                rod_bar_viz = ui.element('div').classes('w-full bg-orange-500 absolute top-0 left-0 shadow-[0_0_10px_orange] transition-all duration-300')
                                rod_bar_viz.style('height: 20%') 
                            ui.label('0%').classes('text-[9px] text-gray-500 mt-1')

                # === RIGHT COL: LOGS & TRENDS ===
                with ui.column().classes('h-full gap-4'):
                    # Trends
                    with ui.card().classes('h-1/2 w-full bg-gray-900/50 border border-gray-800 flex flex-col p-2'):
                        ui.label(T('chart_trend')).classes('text-xs font-bold text-gray-500 mb-2 uppercase')
                        trend_plot = ui.plotly({
                            'layout': {
                                'paper_bgcolor': 'rgba(0,0,0,0)',
                                'plot_bgcolor': 'rgba(0,0,0,0)',
                                'font': {'color': '#94a3b8', 'size': 10},
                                'margin': {'l': 30, 'r': 10, 't': 10, 'b': 30},
                                'xaxis': {'showgrid': False},
                                'yaxis': {'showgrid': True, 'gridcolor': '#334155'},
                                'showlegend': False
                            }
                        }).classes('w-full flex-grow')

                    # Log Console
                    with ui.card().classes('h-1/2 w-full bg-black border border-gray-800 flex flex-col p-0'):
                        ui.label(T('lbl_logs')).classes('text-xs font-bold text-gray-500 p-2 border-b border-gray-800 bg-gray-900')
                        log_container = ui.scroll_area().classes('flex-grow w-full p-2 font-mono text-xs gap-1 flex flex-col')

            # === Simulation Loop (Background) ===
            async def update_loop():
                # Physics
                dt_ui = 0.1
                dt_phys = 0.02
                steps = int(dt_ui / dt_phys)
                
                # Sync Controls from UI to Sim (for safety)
                if sim.control.auto_power:
                    slider_rods.value = sim.rod_position * 100.0
                if sim.control.auto_boron:
                    boron_input.value = sim.boron_concentration

                for _ in range(steps):
                    state = sim.step(dt=dt_phys, flow_factor=sim.thermal.flow_factor)

                # Data Buffers
                now = sim.time
                p_mw = state['total_power']
                t_fuel = state['max_fuel_temp']
                
                time_hist.append(now)
                power_hist.append(p_mw)
                if len(time_hist) > max_history:
                    time_hist.pop(0)
                    power_hist.pop(0)

                # UI Updates
                lbl_power.set_text(f"{p_mw:.0f}")
                lbl_temp.set_text(f"{t_fuel:.0f}")
                lbl_void.set_text(f"{state['max_void_fraction']*100:.1f}")
                
                # Rod Bar
                rod_bar_viz.style(f"height: {sim.rod_position * 100}%")

                # Alarm System
                if t_fuel > 1200:
                    alarm_lamp.classes(replace='bg-red-600 shadow-[0_0_20px_red] animate-pulse')
                    status_label.set_text(T('status_crit'))
                    status_label.classes(replace='text-red-500')
                    # Add log only if not recently added (debounce)
                    if not system_logs or (system_logs[0]['level'] != 'critical' and 'TEMP' not in system_logs[0]['msg']):
                        add_log(T('msg_overheat'), 'critical')
                else:
                    alarm_lamp.classes(replace='bg-green-500 shadow-[0_0_15px_rgba(74,222,128,0.8)]')
                    status_label.set_text(T('status_norm'))
                    status_label.classes(replace='text-green-400')

                # Charts
                z_nodes = np.arange(50)
                core_plot.update_figure({
                    'data': [
                        {
                            'type': 'scatter',
                            'x': z_nodes,
                            'y': state['flux_profile'] * (60.0/1e13), 
                            'name': T('legend_power'),
                            'fill': 'tozeroy',
                            'line': {'color': '#f59e0b', 'width': 3}
                        },
                        {
                            'type': 'scatter',
                            'x': z_nodes,
                            'y': state['fuel_temp'],
                            'name': T('legend_tf'),
                            'yaxis': 'y2',
                            'line': {'color': '#ef4444', 'width': 2}
                        }
                    ]
                })
                
                trend_plot.update_figure({
                    'data': [{
                        'type': 'scatter',
                        'x': list(time_hist),
                        'y': list(power_hist),
                        'mode': 'lines',
                        'line': {'color': '#f59e0b', 'width': 2},
                        'fill': 'tozeroy',
                        'fillcolor': 'rgba(245, 158, 11, 0.1)'
                    }]
                })

                # Logs
                log_container.clear()
                with log_container:
                    for log in system_logs:
                        color = 'text-gray-400'
                        if log['level'] == 'warning': color = 'text-orange-400'
                        if log['level'] == 'critical': color = 'text-red-500 font-bold'
                        ui.label(f"[{log['time']}] {log['msg']}").classes(f"{color} font-mono leading-tight")

            ui.timer(0.1, update_loop)

    render_ui()

ui.run(title='Atomic Sim Pro', dark=True, port=8081, reload=False)
