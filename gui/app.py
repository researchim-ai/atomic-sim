
import sys
import os
from datetime import datetime
from nicegui import ui
import plotly.graph_objects as go
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reactor import ReactorSimulator1D
from reactor.reactor_configs import VVER_1000, RBMK_1000, ReactorConfig, REACTORS

# === Localization ===
LANG = {
    'en': {
        'title': 'ATOMIC SIM',
        'subtitle': 'Reactor Physics Playground',
        'btn_start': 'LAUNCH SIMULATION',
        'btn_setup': 'CONFIGURE & START',
        'select_mode': 'Select Experiment Mode',
        'mode_manual': 'Manual Control',
        'desc_manual': 'Standard operational mode. Control rods and boron manually.',
        'mode_xenon': 'Xenon Oscillations',
        'desc_xenon': 'Demonstrates spatial instability caused by Xe-135.',
        'mode_accident': 'LOCA / Boiling',
        'desc_accident': 'Loss of coolant flow and void fraction feedback.',
        'back_home': 'Exit to Menu',
        'status_norm': 'SYSTEM NORMAL',
        'status_crit': 'CRITICAL WARNING',
        'ctrl_room': 'CONTROL ROOM',
        'tab_main': 'Main',
        'tab_safety': 'Safety',
        'tab_auto': 'Auto',
        'tab_plant': 'Turbine',
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
        'kpi_turbine': 'Electric Output',
        'kpi_sg': 'SG Pressure',
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
        'lbl_rods_short': 'RODS',
        'lbl_ai_advice': 'AI COPILOT ADVICE',
        'lbl_setup_title': 'Mission Configuration',
        'lbl_init_boron': 'Initial Boron (ppm)',
        'lbl_init_rods': 'Initial Rod Position (%)',
        'lbl_specs': 'REACTOR SPECS',
        'spec_type': 'Type',
        'spec_power': 'Nominal Power',
        'spec_fuel': 'Fuel',
        'spec_rods': 'Control Rods',
        'spec_coolant': 'Coolant',
        'lbl_reactor_type': 'Reactor Model',
        'lbl_custom_settings': 'Custom Parameters',
        'lbl_power_mw': 'Thermal Power (MW)',
        'lbl_height_m': 'Core Height (m)',
        'lbl_void_coeff': 'Void Coefficient (Safety!)',
        'lbl_turbine_ctrl': 'Turbine Governor',
        'lbl_feedwater': 'Feedwater Pump'
    },
    'ru': {
        'title': 'АТОМНЫЙ СИМУЛЯТОР',
        'subtitle': 'Песочница реакторной физики',
        'btn_start': 'НАСТРОЙКА ЗАПУСКА',
        'btn_setup': 'НАЧАТЬ СИМУЛЯЦИЮ',
        'select_mode': 'Выберите режим эксперимента',
        'mode_manual': 'Ручное Управление',
        'desc_manual': 'Стандартный режим. Управление стержнями и бором.',
        'mode_xenon': 'Ксеноновые Колебания',
        'desc_xenon': 'Демонстрация пространственной неустойчивости Xe-135.',
        'mode_accident': 'Авария Охлаждения (LOCA)',
        'desc_accident': 'Остановка насоса и кипение теплоносителя.',
        'back_home': 'Выход в Меню',
        'status_norm': 'НОРМАЛЬНАЯ РАБОТА',
        'status_crit': 'КРИТИЧЕСКАЯ ТРЕВОГА',
        'ctrl_room': 'ПУЛЬТ УПРАВЛЕНИЯ',
        'tab_main': 'Главная',
        'tab_safety': 'Защита',
        'tab_auto': 'Авто',
        'tab_plant': 'Турбина',
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
        'kpi_turbine': 'Электрич. Мощность',
        'kpi_sg': 'Давление ПГ',
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
        'lbl_rods_short': 'СУЗ',
        'lbl_ai_advice': 'СОВЕТЫ ИИ-АССИСТЕНТА',
        'lbl_setup_title': 'Конфигурация Миссии',
        'lbl_init_boron': 'Начальный Бор (ppm)',
        'lbl_init_rods': 'Начальное положение стержней (%)',
        'lbl_specs': 'ХАРАКТЕРИСТИКИ РЕАКТОРА',
        'spec_type': 'Тип',
        'spec_power': 'Мощность',
        'spec_fuel': 'Топливо',
        'spec_rods': 'Стержни СУЗ',
        'spec_coolant': 'Теплоноситель',
        'lbl_reactor_type': 'Модель Реактора',
        'lbl_custom_settings': 'Параметры конструктора',
        'lbl_power_mw': 'Тепловая Мощность (МВт)',
        'lbl_height_m': 'Высота Активной Зоны (м)',
        'lbl_void_coeff': 'Паровой Коэффициент (Безопасность!)',
        'lbl_turbine_ctrl': 'Клапан Турбины',
        'lbl_feedwater': 'Питательный Насос'
    }
}

# Global settings
current_lang = 'ru'
session_params = {} # Store init params here

def T(key):
    return LANG[current_lang].get(key, key)

# === Dashboard Page (Menu) ===
@ui.page('/')
def dashboard():
    # Theme setup
    ui.colors(primary='#3b82f6', secondary='#64748b', accent='#f59e0b', dark='#0f172a')
    ui.query('body').style('background-color: #0b0f19; color: #e2e8f0; font-family: "Inter", sans-serif;')

    with ui.column().classes('w-full h-screen items-center justify-center gap-8'):
        # Hero Section
        with ui.column().classes('items-center text-center'):
            ui.icon('science', size='5rem').classes('text-blue-500 mb-4 animate-bounce')
            ui.label(T('title')).classes('text-6xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-600')
            ui.label(T('subtitle')).classes('text-xl text-gray-400 font-light tracking-widest')
            
            # Language Switcher
            with ui.row().classes('mt-4 gap-2'):
                def switch_lang(l):
                    global current_lang
                    current_lang = l
                    ui.navigate.reload()
                
                ui.button('RU', on_click=lambda: switch_lang('ru')).props(f'flat {"color=blue" if current_lang=="ru" else "color=grey"}')
                ui.label('|').classes('text-gray-600 self-center')
                ui.button('EN', on_click=lambda: switch_lang('en')).props(f'flat {"color=blue" if current_lang=="en" else "color=grey"}')

        # Cards Grid
        ui.label(T('select_mode')).classes('text-lg font-bold text-gray-500 mt-8 uppercase tracking-widest')
        
        with ui.grid(columns=3).classes('w-full max-w-5xl gap-6 px-4'):
            
            def create_mode_card(mode_key, desc_key, route, color, icon):
                with ui.card().classes('bg-gray-900/50 border border-gray-800 hover:border-blue-500 transition-all cursor-pointer hover:scale-105 h-full flex flex-col justify-between') \
                        .on('click', lambda: ui.navigate.to(route)):
                    with ui.column().classes('gap-2'):
                        with ui.row().classes('items-center gap-4 mb-2'):
                            ui.icon(icon, size='2rem').classes(f'text-{color}-400')
                            ui.label(T(mode_key)).classes('text-xl font-bold text-gray-200')
                        ui.label(T(desc_key)).classes('text-sm text-gray-400 leading-relaxed')
                    ui.button(T('btn_start'), icon='tune').classes(f'w-full mt-4 bg-{color}-600 hover:bg-{color}-700')

            create_mode_card('mode_manual', 'desc_manual', '/setup/manual', 'blue', 'tune')
            create_mode_card('mode_xenon', 'desc_xenon', '/setup/xenon', 'purple', 'waves')
            create_mode_card('mode_accident', 'desc_accident', '/setup/accident', 'red', 'warning')

# === Setup Page ===
@ui.page('/setup/{mode}')
def setup_page(mode: str):
    ui.colors(primary='#3b82f6', secondary='#64748b', accent='#f59e0b', dark='#0f172a')
    ui.query('body').style('background-color: #0b0f19; color: #e2e8f0; font-family: "Inter", sans-serif;')

    # Defaults
    defaults = {
        'manual': {'boron': 1000.0, 'rods': 20.0},
        'xenon': {'boron': 800.0, 'rods': 40.0},
        'accident': {'boron': 1000.0, 'rods': 10.0}
    }
    init_vals = defaults.get(mode, defaults['manual'])

    with ui.column().classes('w-full h-screen items-center justify-center'):
        with ui.card().classes('w-[600px] bg-gray-900 border border-gray-700 p-8 gap-6'):
            
            with ui.row().classes('items-center gap-4 mb-2'):
                ui.button(icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat round color=white')
                ui.label(T('lbl_setup_title')).classes('text-2xl font-bold text-blue-400')
            
            ui.badge(T(f'mode_{mode}')).classes('mb-4 text-lg')

            # Reactor Selection
            ui.label(T('lbl_reactor_type')).classes('text-gray-400 font-bold')
            
            reactor_options = {
                'vver': 'VVER-1000 (Standard PWR)',
                'rbmk': 'RBMK-1000 (Channel Type LWGR)',
                'custom': 'Custom (Experimental)'
            }
            
            selected_reactor = ui.select(reactor_options, value='vver').classes('w-full mb-4').props('outlined behavior="menu"')
            
            # Custom Settings (Collapsible)
            with ui.expansion(T('lbl_custom_settings'), icon='build').classes('w-full bg-gray-800/50 rounded border border-gray-700').bind_visibility_from(selected_reactor, 'value', lambda v: v == 'custom'):
                with ui.column().classes('p-4 w-full'):
                    ui.label(T('lbl_power_mw')).classes('text-xs text-gray-500')
                    custom_power = ui.slider(min=1000, max=5000, value=3000).props('label-always color=purple')
                    
                    ui.label(T('lbl_height_m')).classes('text-xs text-gray-500')
                    custom_height = ui.slider(min=2.0, max=10.0, value=3.5, step=0.1).props('label-always color=blue')
                    
                    ui.label(T('lbl_void_coeff')).classes('text-xs text-gray-500')
                    # Void coeff: -0.2 (safe) to +0.05 (dangerous)
                    custom_void = ui.slider(min=-0.2, max=0.05, value=-0.05, step=0.01).props('label-always color=red')

            ui.separator().classes('bg-gray-700')

            # Initial Conditions
            ui.label(T('lbl_init_boron')).classes('text-gray-400 font-bold')
            boron_slider = ui.slider(min=0, max=2000, value=init_vals['boron']).props('label-always color=cyan')
            
            ui.label(T('lbl_init_rods')).classes('text-gray-400 font-bold mt-4')
            rods_slider = ui.slider(min=0, max=100, value=init_vals['rods']).props('label-always color=orange')

            def start():
                # Create config if custom
                r_type = selected_reactor.value
                
                if r_type == 'custom':
                    # Create temp custom config
                    conf = ReactorConfig(
                        name="Custom Reactor",
                        type_str="Experimental",
                        thermal_power=custom_power.value,
                        core_height=custom_height.value,
                        num_assemblies=100,
                        beta_eff=0.0065,
                        neutron_speed=1000.0,
                        temp_coeff_fuel=-2e-5,
                        temp_coeff_coolant=-5e-4,
                        void_coeff=custom_void.value,
                        scram_speed=0.1,
                        shim_speed=0.01
                    )
                    REACTORS['custom_session'] = conf
                    r_type = 'custom_session'

                # Save params
                session_params[mode] = {
                    'boron': boron_slider.value,
                    'rods': rods_slider.value / 100.0,
                    'reactor': r_type
                }
                ui.navigate.to(f'/run/{mode}')

            ui.button(T('btn_setup'), icon='rocket_launch', on_click=start).classes('w-full bg-green-600 hover:bg-green-700 text-lg font-bold mt-6 py-4')


# === Simulation Runner Page ===
@ui.page('/run/{mode}')
def run_simulation(mode: str):
    # 1. Initialize Simulator based on Mode and User Params
    
    # Load params
    params = session_params.get(mode, {'boron': 1000.0, 'rods': 0.2, 'reactor': 'vver'})
    
    # Init Simulator with chosen reactor config
    sim = ReactorSimulator1D(config_name=params['reactor'], num_nodes=50, device='cpu')
    
    sim.boron_concentration = params['boron']
    sim.control.rod_position = params['rods']
    sim.control.boron_concentration = params['boron'] # Sync control object state
    
    # Pre-stabilization
    for _ in range(50):
        sim.step(dt=0.01)

    # 2. Session State (Logs, History)
    max_history = 300
    time_hist = []
    power_hist = []
    system_logs = []

    def add_log(msg, level='info'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {'time': timestamp, 'msg': msg, 'level': level}
        system_logs.insert(0, entry)
        if len(system_logs) > 50: system_logs.pop()

    # 3. Build UI
    ui.colors(primary='#3b82f6', secondary='#64748b', accent='#f59e0b', dark='#0f172a')
    ui.query('body').style('background-color: #0b0f19; color: #e2e8f0; font-family: "Inter", sans-serif;')

    # UI Refs dictionary to avoid scoping issues
    refs = {}

    # --- Header ---
    with ui.header().classes('bg-gray-900 items-center justify-between elevation-4 q-py-sm'):
        with ui.row().classes('items-center gap-4'):
            ui.button(icon='arrow_back', on_click=lambda: ui.navigate.to('/')).props('flat round color=white') \
                .tooltip(T('back_home'))
            
            ui.icon('science', size='2rem').classes('text-blue-400')
            ui.label(T('title')).classes('text-xl font-bold tracking-wider text-blue-100')
            
            # Mode Badge
            mode_color = {'manual': 'blue', 'xenon': 'purple', 'accident': 'red'}.get(mode, 'gray')
            ui.badge(T(f'mode_{mode}'), color=f'{mode_color}-800').classes('text-xs')

        # Status & Time
        with ui.row().classes('items-center gap-6'):
            refs['alarm_lamp'] = ui.html('<div class="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_10px_rgba(74,222,128,0.8)]"></div>', sanitize=False)
            refs['status_label'] = ui.label(T('status_norm')).classes('text-green-400 font-mono font-bold')
            ui.label('TIME:').classes('font-mono text-gray-500 text-xs')
            ui.label('0.0s').classes('font-mono text-blue-300 min-w-[60px]').bind_text_from(sim, 'time', lambda t: f"{t:.1f}s")

    # --- Main Content ---
    with ui.grid(columns='350px 1fr 300px').classes('w-full h-[calc(100vh-60px)] gap-4 p-4'):
        
        # === LEFT: CONTROLS ===
        with ui.card().classes('h-full bg-gray-900/50 border border-gray-800 flex flex-col gap-4'):
            ui.label(T('ctrl_room')).classes('text-lg font-bold text-blue-400 border-b border-gray-800 pb-2')
            
            with ui.tabs().classes('w-full text-blue-400') as tabs:
                main_tab = ui.tab(T('tab_main'))
                safety_tab = ui.tab(T('tab_safety'))
                auto_tab = ui.tab(T('tab_auto'))
                plant_tab = ui.tab(T('tab_plant'))

            with ui.tab_panels(tabs, value=main_tab).classes('w-full bg-transparent flex-grow'):
                
                # MAIN TAB
                with ui.tab_panel(main_tab).classes('flex flex-col gap-6'):
                    # Rods
                    with ui.column().classes('w-full'):
                        ui.label(T('lbl_rods')).classes('text-sm text-gray-400 font-bold')
                        slider_rods = ui.slider(min=0, max=100, value=sim.rod_position*100, step=0.1).props('label-always color=orange track-size=8px thumb-size=20px')
                        def manual_rods(e):
                            if not sim.control.auto_power:
                                sim.control.rod_position = slider_rods.value / 100.0
                        slider_rods.on('change', manual_rods)
                        ui.linear_progress(value=0.2).bind_value_from(sim.control, 'rod_position').props('color=orange track-color=gray-800 size=15px rounded')

                    # Boron
                    with ui.column().classes('w-full'):
                        ui.label(T('lbl_boron')).classes('text-sm text-gray-400 font-bold')
                        def set_boron(e):
                            try: sim.control.boron_concentration = float(boron_input.value)
                            except: pass
                        boron_input = ui.number(value=sim.control.boron_concentration, format='%.0f', on_change=set_boron).classes('w-full text-lg').props('standout outlined dense')
                        ui.slider(min=0, max=2000, value=1000).bind_value(boron_input).props('color=cyan')

                # AUTO TAB
                with ui.tab_panel(auto_tab).classes('flex flex-col gap-4'):
                    ui.label(T('lbl_auto_sys')).classes('font-bold text-center mb-2')
                    sw_p = ui.switch(T('sw_pid'), value=sim.control.auto_power).classes('text-green-400')
                    sw_b = ui.switch(T('sw_shim'), value=sim.control.auto_boron).classes('text-cyan-400')
                    
                    ui.separator().classes('bg-gray-800')
                    ui.label(T('lbl_target') + ' (MW)').classes('text-sm text-gray-400')
                    target_p = ui.number(value=sim.control.target_power, step=100).classes('w-full').props('outlined dense suffix="MW"')
                    
                    def update_auto():
                        sim.control.auto_power = sw_p.value
                        sim.control.auto_boron = sw_b.value
                        sim.control.target_power = target_p.value
                        slider_rods.disable() if sw_p.value else slider_rods.enable()
                        boron_input.disable() if sw_b.value else boron_input.enable()
                    
                    sw_p.on('update:model-value', update_auto)
                    sw_b.on('update:model-value', update_auto)
                    target_p.on('change', update_auto)

                # PLANT TAB (Secondary)
                with ui.tab_panel(plant_tab).classes('flex flex-col gap-4'):
                     ui.label('TURBINE & SECONDARY').classes('text-cyan-400 font-bold text-center mb-2')
                     
                     ui.label(T('lbl_turbine_ctrl')).classes('text-sm text-gray-400')
                     slider_turbine = ui.slider(min=0, max=100, value=100).props('label-always color=cyan track-size=8px')
                     def set_turbine(e):
                         sim.plant.turbine_throttle = slider_turbine.value / 100.0
                     slider_turbine.on('change', set_turbine)
                     
                     ui.label(T('lbl_feedwater')).classes('text-sm text-gray-400')
                     slider_fw = ui.slider(min=0, max=100, value=100).props('label-always color=teal track-size=8px')
                     def set_fw(e):
                         sim.plant.feedwater_pump_speed = slider_fw.value / 100.0
                     slider_fw.on('change', set_fw)

                # SAFETY TAB
                with ui.tab_panel(safety_tab).classes('flex flex-col gap-4'):
                    ui.label(T('lbl_emergency')).classes('text-red-500 font-bold')
                    def scram():
                        sim.control.rod_position = 1.0
                        slider_rods.value = 100
                        sw_p.value = False
                        sim.control.auto_power = False
                        add_log(T('msg_scram'), 'critical')
                    ui.button(T('btn_scram'), on_click=scram).classes('bg-red-600 w-full font-bold py-4 text-lg shadow-[0_0_15px_rgba(220,38,38,0.6)] hover:bg-red-700 transition-all')

                    ui.separator().classes('bg-gray-800 my-2')
                    ui.label(T('lbl_faults')).classes('text-orange-400 font-bold')
                    
                    def trip_pump():
                        sim.pump_signal = 0.1
                        add_log(T('msg_pump'), 'warning')
                    ui.button(T('btn_trip'), on_click=trip_pump).props('outline color=orange').classes('w-full')
                    
                    def eject_rod():
                        sim.control.rod_position = 0.0
                        slider_rods.value = 0
                        add_log(T('msg_rod'), 'critical')
                    ui.button(T('btn_eject'), on_click=eject_rod).props('outline color=red').classes('w-full')

        # === CENTER: VIZ ===
        with ui.column().classes('h-full gap-4'):
            # KPIs
            with ui.row().classes('w-full h-[100px] gap-2'):
                def kpi_card(title, unit, color):
                    with ui.card().classes(f'flex-1 h-full bg-gray-900/50 border-l-4 border-{color}-500 flex flex-col justify-center items-center p-1'):
                        ui.label(title).classes('text-gray-400 text-[10px] uppercase tracking-widest text-center')
                        lbl = ui.label('0').classes(f'text-2xl font-mono font-bold text-{color}-400')
                        ui.label(unit).classes(f'text-{color}-600 text-xs font-bold')
                        return lbl
                refs['lbl_power'] = kpi_card(T('kpi_power'), 'MW', 'orange')
                refs['lbl_temp'] = kpi_card(T('kpi_temp'), '°C', 'red')
                refs['lbl_turbine'] = kpi_card(T('kpi_turbine'), 'MW', 'cyan')
                refs['lbl_sg_press'] = kpi_card(T('kpi_sg'), 'MPa', 'teal')

            # Main Chart
            with ui.row().classes('w-full flex-grow bg-gray-900 border border-gray-800 p-0 overflow-hidden relative no-wrap'):
                refs['core_plot'] = ui.plotly({
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
                
                # Side Rod Bar
                with ui.column().classes('w-[60px] h-full border-l border-gray-700 items-center p-2 bg-gray-800/30'):
                    ui.label(T('lbl_rods_short')).classes('text-[10px] font-bold text-gray-400 mb-1')
                    with ui.element('div').classes('w-[20px] flex-grow bg-gray-700 rounded relative overflow-hidden'):
                        refs['rod_bar'] = ui.element('div').classes('w-full bg-orange-500 absolute top-0 left-0 shadow-[0_0_10px_orange] transition-all duration-300')
                        refs['rod_bar'].style('height: 20%') 
                    ui.label('0%').classes('text-[9px] text-gray-500 mt-1')

        # === RIGHT: LOGS ===
        with ui.column().classes('h-full gap-4'):
            # Specs Card
            with ui.card().classes('w-full bg-gray-900/30 border border-gray-700 p-3'):
                ui.label(T('lbl_specs')).classes('text-[10px] font-bold text-gray-500 mb-2 tracking-widest')
                
                # Initial Static Display (will be updated in loop)
                ui.label(f"{T('spec_type')}: {sim.config.name}").classes('text-xs text-blue-300 font-bold')
                ui.label(f"{T('spec_power')}: {sim.config.thermal_power} MW").classes('text-xs text-gray-400')
                ui.label(f"{T('spec_coolant')}: Void Coeff {sim.config.void_coeff}").classes('text-xs text-gray-400')
            
            # AI Copilot
            with ui.card().classes('flex-grow bg-blue-900/20 border border-blue-800 flex flex-col p-2 relative overflow-hidden'):
                ui.icon('smart_toy', size='4rem').classes('absolute -right-4 -bottom-4 text-blue-800 opacity-50')
                ui.label(T('lbl_ai_advice')).classes('text-xs font-bold text-blue-400 mb-2 uppercase tracking-widest')
                refs['ai_text'] = ui.label('Analyzing system parameters...').classes('text-sm text-blue-100 font-mono leading-relaxed z-10')

            with ui.card().classes('h-1/3 w-full bg-gray-900/50 border border-gray-800 flex flex-col p-2'):
                ui.label(T('chart_trend')).classes('text-xs font-bold text-gray-500 mb-2 uppercase')
                refs['trend_plot'] = ui.plotly({
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

            with ui.card().classes('h-1/3 w-full bg-black border border-gray-800 flex flex-col p-0'):
                ui.label(T('lbl_logs')).classes('text-xs font-bold text-gray-500 p-2 border-b border-gray-800 bg-gray-900')
                refs['log_container'] = ui.scroll_area().classes('flex-grow w-full p-2 font-mono text-xs gap-1 flex flex-col')

    # === LOOP ===
    async def update_loop():
        dt_ui = 0.1
        dt_phys = 0.02
        steps = int(dt_ui / dt_phys)
        
        # Sync UI -> Sim
        if sim.control.auto_power:
            slider_rods.value = sim.control.rod_position * 100.0
        if sim.control.auto_boron:
            boron_input.value = sim.control.boron_concentration

        # Physics Steps
        for _ in range(steps):
            state = sim.step(dt=dt_phys)

        # Data
        p_mw = state['thermal_power_mw'] # Real thermal power
        t_fuel = state['max_fuel_temp']
        p_electric = state['turbine_power_mw']
        p_sg = state['sg_pressure'] / 1e6 # MPa
        
        time_hist.append(sim.time)
        power_hist.append(p_mw)
        if len(time_hist) > max_history:
            time_hist.pop(0)
            power_hist.pop(0)

        # UI Updates (using refs)
        refs['lbl_power'].set_text(f"{p_mw:.0f}")
        refs['lbl_temp'].set_text(f"{t_fuel:.0f}")
        refs['lbl_turbine'].set_text(f"{p_electric:.0f}")
        refs['lbl_sg_press'].set_text(f"{p_sg:.1f}")
        
        refs['rod_bar'].style(f"height: {sim.control.rod_position * 100}%")

        # Alarms
        if t_fuel > 1200:
            refs['alarm_lamp'].classes(replace='bg-red-600 shadow-[0_0_20px_red] animate-pulse')
            refs['status_label'].set_text(T('status_crit'))
            refs['status_label'].classes(replace='text-red-500')
            if not system_logs or (system_logs[0]['level'] != 'critical' and 'TEMP' not in system_logs[0]['msg']):
                add_log(T('msg_overheat'), 'critical')
        else:
            refs['alarm_lamp'].classes(replace='bg-green-500 shadow-[0_0_15px_rgba(74,222,128,0.8)]')
            refs['status_label'].set_text(T('status_norm'))
            refs['status_label'].classes(replace='text-green-400')

        # Plot Updates
        z_nodes = np.arange(50)
        refs['core_plot'].update_figure({
            'data': [
                {
                    'type': 'scatter', 'x': z_nodes, 'y': state['flux_profile'] * (60.0/1e13), 
                    'name': T('legend_power'), 'fill': 'tozeroy', 'line': {'color': '#f59e0b', 'width': 3}
                },
                {
                    'type': 'scatter', 'x': z_nodes, 'y': state['fuel_temp'], 
                    'name': T('legend_tf'), 'yaxis': 'y2', 'line': {'color': '#ef4444', 'width': 2}
                }
            ]
        })
        
        refs['trend_plot'].update_figure({
            'data': [{'type': 'scatter', 'x': list(time_hist), 'y': list(power_hist), 'mode': 'lines', 'line': {'color': '#f59e0b', 'width': 2}, 'fill': 'tozeroy', 'fillcolor': 'rgba(245, 158, 11, 0.1)'}]
        })

        # Logs
        refs['log_container'].clear()
        with refs['log_container']:
            for log in system_logs:
                color = 'text-gray-400'
                if log['level'] == 'warning': color = 'text-orange-400'
                if log['level'] == 'critical': color = 'text-red-500 font-bold'
                ui.label(f"[{log['time']}] {log['msg']}").classes(f"{color} font-mono leading-tight whitespace-normal break-words")

        # AI Copilot Logic
        advice = []
        if t_fuel > 1000:
            advice.append("⚠️ CRITICAL: Fuel temp high! SCRAM immediately!")
        elif t_fuel > 800:
            advice.append("Warning: Fuel temp rising. Insert rods.")
        
        if state['max_void_fraction'] > 0.1:
             advice.append("Boiling detected!")
             if sim.config.void_coeff > 0:
                 advice.append("DANGER: Positive void coeff!")
        
        # Add Plant advice
        if p_sg > 7.0:
            advice.append("SG Pressure HIGH! Open Turbine Valve.")
             
        if abs(p_mw - sim.control.target_power) > 50 and sim.control.auto_power:
            advice.append("Regulating power to target...")
            
        if not advice:
            advice.append("System stable. Parameters nominal.")
            
        refs['ai_text'].set_text(" ".join(advice))

    ui.timer(0.1, update_loop)

ui.run(title='Atomic Sim', dark=True, port=8081, reload=False)
