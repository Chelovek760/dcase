import numpy as np

import F5signal as f5s
class Equipment():
    """
    The class describing the analyzed object
    """
    def __init__(self, name):
        self.component = {}
        self.name = name
        self.__heuristic_param = {}

    def add_components(self, component_dict):
        
        for i in list(component_dict.keys()):

            if i not in list(heuroistic_component.keys()):
                raise AttributeError ("Отсутствует такой компонент")
            else:
                self.component[i] = component_dict[i]
                self.__heuristic_param[i] = {}

    def heuristic_param(self):
        
        for component in self.__heuristic_param:
            print(component)
            print('-' * len(component))
            for defect_freq in self.__heuristic_param[component]:
                print(defect_freq, self.__heuristic_param[component][defect_freq])

    def get_freq(self):
        return self.__heuristic_param

    def calculate_freq(self, freq):
    
        for component in list(self.component):
            for func in heuroistic_component[component]:
                self.__heuristic_param[component][func.__name__] = func(freq, self.component[component])
                print(f'{func.__name__}, посчитана')
        print('\n')


def freq_rotation_out_ring(freq, param):
    """Функиця расчета частоты перекатывания тел по наружнему кольцу.

    Params:
        freq:              (integer) - Основная частота вращения;
        diam_rolling_body: (integer) - Диаметр тела вращения;
        diam_sep:          (integer) - Диаметр сепаратора;
        count:             (integer) - Количество тел качения;
        rad:               (integer) - Угол контакта тел качения с дорожками качения (Для радиальных подшипников rad = 1, 
                                                                                    Для Упорных подшипников rad = 0);
                                                                                                                                                                
    returns: 
        freq_out: (integer) - частота перекатывания тел по наружнему кольцу.
    """
    freq_out = []
    for par in param[0]:
        diam_rolling_body = par['Диаметр тела качения']
        diam_sep = (par['Диаметр внутренний'] + par['Диаметр наружний']) / 2
        count = par['Количество тел качения']
        rad = par['Угол']
        freq_out.append(round(0.5 * freq * (1 - diam_rolling_body / diam_sep * np.cos(rad)) * count, 2))

    return freq_out

def stator_tooth_frequency(freq, count_tooth_rot=1):
    """Функиця расчета частоты перекатывания тел по внутреннему кольцу.

    Params:
        freq:              (integer) - Основная частота вращения;
        count_tooth_rot:   (integer) - Количево зубцов;                                                                      Для Упорных подшипников rad = 0).

    returns:                (float) - Зубцовая частота

    """
    tooth_frequency = freq * count_tooth_rot
    return tooth_frequency

def freq_rotation_inner_ring(freq, param):
    """Функиця расчета частоты перекатывания тел по внутреннему кольцу.

    Params:
        freq:              (integer) - Основная частота вращения;
        diam_rolling_body: (integer) - Диаметр тела вращения;
        diam_sep:          (integer) - Диаметр сепаратора;
        count:             (integer) - Количество тел качения;
        rad:               (integer) - Угол контакта тел качения с дорожками качения (Для радиальных подшипников rad = 1, 
                                                                                    Для Упорных подшипников rad = 0).

    returns: 
        freq_inner: (integer) - частота перекатывания тел по наружнему кольцу.
    """
    freq_out = []
    for par in param[0]:
        diam_rolling_body = par['Диаметр тела качения']
        diam_sep = (par['Диаметр внутренний'] + par['Диаметр наружний']) / 2
        count = par['Количество тел качения']
        rad = par['Угол']
        freq_out.append(round(0.5 * freq * (1 + diam_rolling_body / diam_sep * np.cos(rad)) * count, 2))

    return freq_out


def freq_rotation_sep(freq, param):
    """Функиця расчета частоты вращения сепаратора.

    Params:
        freq:              (integer) - Основная частота вращения;
        diam_rolling_body: (integer) - Диаметр тела вращения;
        diam_sep:          (integer) - Диаметр сепаратора;
        rad:               (integer) - Угол контакта тел качения с дорожками качения (Для радиальных подшипников cos(rad) = 1, 
                                                                                    Для Упорных подшипников cos(rad) = 0).

    returns: 
        freq_sep: (integer) - частота перекатывания тел по наружнему кольцу.
    """
    freq_out = []
    for par in param[0]:
        diam_rolling_body = par['Диаметр тела качения']
        diam_sep = (par['Диаметр внутренний'] + par['Диаметр наружний']) / 2
        rad = par['Угол']
        freq_out.append(round(0.5 * freq * (1 - diam_rolling_body / diam_sep * np.cos(rad)) , 2))

    return freq_out

def freq_rotation_body(freq, param):
    """Функиця расчета частоты вращения тела качения.

    Params:
        freq:              (integer) - Основная частота вращения;
        diam_rolling_body: (integer) - Диаметр тела вращения;
        diam_sep:          (integer) - Диаметр сепаратора;
        rad:               (integer) - Угол контакта тел качения с дорожками качения (Для радиальных подшипников rad = 1, 
                                                                                    Для Упорных подшипников rad = 0).

    returns: 
        freq_sep: (integer) - частота перекатывания тел по наружнему кольцу.
    """
    freq_out = []
    for par in param[0]:
        diam_rolling_body = par['Диаметр тела качения']
        diam_sep = (par['Диаметр внутренний'] + par['Диаметр наружний']) / 2
        rad = par['Угол']
        freq_out.append(round(0.5 * freq * (diam_sep / diam_rolling_body) * (1 - diam_rolling_body ** 2 / diam_sep ** 2 ** np.cos(rad) ** 2), 2))

    return freq_out

############################################################   ЭВРИСТИКИ ДЛЯ ПОДШИПНИКОВ КАЧЕНИЯ   ############################################################
def shaft_shaking(freq, *param):
    """Описывает дефект боя вала.

    Params:
        freq: (integer) - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main':[], 'Add':[]}
    k = np.array([1, 2, 3, 4])
    freqs['Main'] = k * freq

    return freqs


def uneven_rad_loaded(freq, *param):
    """Описывает дефект неравномерного радиального натяга.
    
    Params:
        freq: (integer) - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main':[], 'Add':[]}
    freqs['Main'] = 2 * freq
    k = np.array([1, 2, 3, 4])
    freqs['Add'] = 2 * k * freq

    return freqs

def swash_out_ring(freq, *param):
    """Описывает дефект перекоса наружнoго кольца.

    Add some param

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    freq_out = freq_rotation_out_ring(freq, param)
    k = np.array([1, 2, 3, 4])
    for i in freq_out:
        freqs['Add'].append(2 * k * i)
        freqs['Main'].append(2 * i) 

    return freqs

def wear_out_ring(freq, *param):
    """Описывает дефект износа наружнoго кольца.

    Add some param

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    freq_out = freq_rotation_out_ring(freq, param)
    freqs['Main'] = freq_out
    k = np.array([1, 2, 3])
    for i in freq_out:
        freqs['Add'].append(k * i)
    
    return freqs

def sink_out_ring(freq, *param):
    """Описывает дефект раковин наружнего колеса

    CHECK!!! Add some param

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    freq_rotat = freq_rotation_out_ring(freq, param)
    k = np.array([4, 5, 6])
    for i in freq_rotat:
        freqs['Main'].append(k * i)

    return freqs


def wear_inner_ring(freq, *param):
    """Описывает дефект износа внутренного кольца

    Add some param

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4])
    freqs['Main'] = k * freq
    freq_rotor_inner_ring = freq_rotation_inner_ring(freq, param)
    for i in freq_rotor_inner_ring:
        freqs['Add'].append(i)

    return freqs

def sink_inner_ring(freq, *param):
    """Описывает дефект раковин на внутреннем колеце

    Add some param

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    freq_rotation = freq_rotation_inner_ring(freq, param)
    k = np.array([1, 2, 3, 4])
    for i in freq_rotation:
        freqs['Main'].append(k * i)
    freqs['Add'].append(k * freq)

    return freqs

def weak_sep(freq, *param):
    """Описывает дефект износа тела качения и сепаратора


    Add some param

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    freq_rotation = freq_rotation_sep(freq, param)
    k = np.array([1, 2, 3, 4]) 
    for i in freq_rotation:
        freqs['Main'].append(i)
        freqs['Main'].append(freq - i)
        freqs['Add'].append(k * i)
        freqs['Add'].append(k * (freq - i))

    return freqs

def break_inner_body(freq, *param):
    """Описывает дефект раковин, сколов на теле качения

    Add some param

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4])
    freq = freq_rotation_body(freq, param)
    for i in freq:
        freqs['Main'].append(2 * k * i)

    return freqs

# def defect_mount_fitting(freq, *param):
#     """Описывает дефект узлов крепления

#     ОТНОСИТЕЛЬНАЯ ЭВРИСТИКА

#     Params:
#         freq: integer - Основная частота вращения.

#     returns: 
#         freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

#                             'Add:{Гармоника основной частоты: Список частот}}.
#     """
#     freqs = {'Main': [], 'Add': []}
#     k = np.array([0.5, 0.25, 0.125])
#     freqs['Main'].append(k * freq)

#     return freqs

def defect_clutch(freq, *param):
    """Описывает дефект муфты

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    k = np.array([8, 9, 10, 11])
    freqs['Main'].append(k * freq)

    return freqs
############################################################# Агрегаты с рабочими колесами ###############################################################


def shaft_working_shaking(freq):
    """Описывает дефект боя рабочего колеса

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4])
    freqs['Main'].append(k * freq)

    return freqs


def defect_slave_fittings(freq):
    """Описывает дефект узлов крепления

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    k = np.array([0.5, 0.25, 0.125])
    freqs['Main'].append(k * freq)

    return freqs

def autocolleb_work_wheel(freq):
    """Описывает дефект автоколебания рабочего колеса

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4])
    freqs['Main'].append(k / 2 * freq)
    freqs['Main'].append(k / 3 * freq)

    return freqs

def defect_blade(freq, param):
    """Описывает дефект автоколебания рабочего колеса

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4])
    freqs['Main'].append(k * freq)
    freqs['Main'].append(k / 3 * freq)
    N = param['Вентилятор']['Количество лопастей']

    return freqs

def inhomogenuity_flow(freq, param):
    """Описывает дефект неоднородности потока

    Params:
        freq: integer - Основная частота вращения.

    returns: 
        freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                            'Add:{Гармоника основной частоты: Список частот}}.
    """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4])
    N = param['Вентилятор']['Количество лопастей']

    freqs['Main'].append(K * N * freq)


    return freqs


############################################################   ЭВРИСТИКИ ДЛЯ АСИНХР. ДВИГ.   ############################################################

def stator_winding_defects_async(freq, *param):
    """Дефект обмоток статора

        Add some param

        Params:
            freq: integer - Частота питющего напряжения.

        returns:
            freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                'Add:{Гармоника основной частоты: Список частот}}.
        """

    freqs = {'Main': [], 'Add': []}
    k = np.array([2, 3, 4, 5, 6])
    freqs['Main'].append(2 * freq)
    freqs['Add'].append(2 * k + 2 * freq)

    return freqs


def dynamic_eccentricity_gap_async(freq, *param):
    """Динамический эксцентриситет с насыщением зубов

            Add some param

            Params:
                freq: integer - Частота вращения ротора.

            returns:
                freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                    'Add:{Гармоника основной частоты: Список частот}}.
            """
    freqs = {'Main': [], 'Add': []}
    count_tooth_rot = param['Количество зубьев ротора']
    count_tooth_stat = param['Количество зубьев статора']
    power_freq = param['Частота питающего напряжения']
    k = np.array([1, 2, 3, 4, 5, 6])
    freqs['Main'] = [freq, 2 * freq, 2 * power_freq - freq, 2 * power_freq + freq]
    for i in k:
        freqs['Add'].append(i * stator_tooth_frequency(power_freq, count_tooth_rot) + i * freq)
        freqs['Add'].append(i * stator_tooth_frequency(power_freq, count_tooth_rot) - i * freq)
        freqs['Add'].append(i * stator_tooth_frequency(power_freq, count_tooth_stat))
    return freqs


def nonlinear_voltage_distortion(freq, *param):
    """Нелинейные искажения напряжжения

            Add some param

            Params:
                freq: integer - Частота питающего напряжения.

            returns:
                freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                    'Add:{Гармоника основной частоты: Список частот}}.

            """
    freqs = {'Main': [], 'Add': []}
    k = 6 * np.array([2, 3, 4, 5, 6])
    freqs['Main'] = freq * k
    return freqs


############################################################   ЭВРИСТИКИ РЕДУКТОРА  ############################################################
def thooth__freq(freq, count_gear):
    return freq * count_gear


def gear_shaft_impact(freq, **param):
    """Бой Ведущего вала

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4, 5])
    kadd = np.array([1, 2, 3])
    count_thooth = param['Количество зубьев']
    for harm in k:
        freqs['Main'].append(harm * freq)
        freqs['Main'].append(harm * thooth__freq(freq, count_thooth) - harm * freq)
        freqs['Main'].append(harm * thooth__freq(freq, count_thooth) + harm * freq)
    freqs['Add'] = kadd * freq
    return freqs


def gear_skew(freq, **param):
    """Перекос  шестерни

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    freqs = {'Main': [], 'Add': []}
    k = 2 * np.array([1, 2, 3, 4, 5])
    kadd = 2 * np.array([1, 2, 3])
    count_thooth = param['Количество зубьев']
    for harm in k:
        freqs['Main'].append(harm * freq)
        freqs['Main'].append(harm * thooth__freq(freq, count_thooth) - freq)
        freqs['Main'].append(harm * thooth__freq(freq, count_thooth) + freq)
    freqs['Add'] = kadd * freq
    return freqs


def fault_thooth_vedushego(freq, **param):
    """Дефект зубъев

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4])
    kadd = np.array([4, 5, 6, 7])
    count_thooth = param['Количество зубьев']
    for harm in k:
        freqs['Main'].append(harm * freq)
        freqs['Main'].append(harm * thooth__freq(freq, count_thooth) - harm * freq)
        freqs['Main'].append(harm * thooth__freq(freq, count_thooth) + harm * freq)
    freqs['Add'] = kadd * freq
    return freqs


def fault_gearing(freq, **param):
    """Дефект зацепления или смазки

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    freqs = {'Main': [], 'Add': []}
    k = np.array([1, 2, 3, 4])
    count_thooth = param['Количество зубьев']
    for harm in k:
        freqs['Main'].append(harm * thooth__freq(freq, count_thooth))
    return freqs


############################################################   ЭВРИСТИКИ РЕМЕННОЙ ПЕРЕДАЧИ  ############################################################

def freq_rot_vedomogo(freq, diameter_first=1, diameter_second=1):
    return freq * diameter_first / diameter_second


def freq_rot_belt(freq, diameter_first=1, len_belt=1):
    return freq * diameter_first * np.pi / len_belt


def gear_shaft_first_belt(freq, **param):
    """Бой ведущего вала

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    diameter_first = param['Диаметр ведущего']
    diameter_second = param['Диаметр ведомого']
    c_r_v = freq_rot_vedomogo(freq, diameter_first, diameter_second)
    main_harm_boy_vedushego_vala = 1
    freqs = {'Main': [freq * c_r_v * main_harm_boy_vedushego_vala - freq,
                      freq * c_r_v * main_harm_boy_vedushego_vala + freq],
             'Add': [freq * c_r_v * main_harm_boy_vedushego_vala - freq,
                     freq * c_r_v * main_harm_boy_vedushego_vala + freq]}
    return freqs


def gear_shaft_second_belt(freq, **param):
    """Бой ведомого вала

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    diameter_first = param['Диаметр ведущего']
    diameter_second = param['Диаметр ведомого']
    c_r_v = freq_rot_vedomogo(freq, diameter_first, diameter_second)
    main_harm_boy_vedushego_vala = 1
    freqs = {'Main': [freq * c_r_v * main_harm_boy_vedushego_vala - c_r_v,
                      freq * c_r_v * main_harm_boy_vedushego_vala + c_r_v],
             'Add': [freq * c_r_v * main_harm_boy_vedushego_vala - c_r_v,
                     freq * c_r_v * main_harm_boy_vedushego_vala + c_r_v]}
    return freqs


def first_gear_skew(freq, **param):
    """Перекос ведущего шкива

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    diameter_first = param['Диаметр ведущего']
    diameter_second = param['Диаметр ведомого']
    c_r_v = freq_rot_vedomogo(freq, diameter_first, diameter_second)
    main_harm_boy_vedushego_vala = 1
    freqs = {'Main': [2 * (freq * c_r_v * main_harm_boy_vedushego_vala - freq),
                      2 * (freq * c_r_v * main_harm_boy_vedushego_vala + freq)],
             'Add': [2 * (freq * c_r_v * main_harm_boy_vedushego_vala - freq),
                     2 * (freq * c_r_v * main_harm_boy_vedushego_vala + freq)]}
    return freqs


def second_gear_skew(freq, **param):
    """Перекос ведомго шкива

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    diameter_first = param['Диаметр ведущего']
    diameter_second = param['Диаметр ведомого']
    c_r_v = freq_rot_vedomogo(freq, diameter_first, diameter_second)
    main_harm_boy_vedushego_vala = 1
    freqs = {'Main': [2 * (freq * c_r_v * main_harm_boy_vedushego_vala - c_r_v),
                      2 * (freq * c_r_v * main_harm_boy_vedushego_vala + c_r_v)],
             'Add': [2 * (freq * c_r_v * main_harm_boy_vedushego_vala - c_r_v),
                     2 * (freq * c_r_v * main_harm_boy_vedushego_vala + c_r_v)]}
    return freqs


def deterioration_first_pulley(freq, **param):
    """Износ ведущего ремня

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.
                """
    diameter_first = param['Диаметр ведущего']
    diameter_second = param['Диаметр ведомого']
    c_r_v = freq_rot_vedomogo(freq, diameter_first, diameter_second)
    main_harm = [1, 2, 3, 4, 5, 6]
    freqs = {'Main': [], 'Add': []}
    for main_harm_boy_vedushego_vala in main_harm:
        freqs['Main'].append(freq * c_r_v * main_harm_boy_vedushego_vala - freq)
        freqs['Main'].append(freq * c_r_v * main_harm_boy_vedushego_vala + freq)
        freqs['Add'].append(freq * c_r_v * main_harm_boy_vedushego_vala - freq)
        freqs['Add'].append(freq * c_r_v * main_harm_boy_vedushego_vala + freq)
    return freqs


def deterioration_second_pulley(freq, **param):
    """Износ веломого ремня

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    diameter_first = param['Диаметр ведущего']
    diameter_second = param['Диаметр ведомого']
    c_r_v = freq_rot_vedomogo(freq, diameter_first, diameter_second)
    main_harm = [1, 2, 3, 4, 5, 6]
    freqs = {'Main': [], 'Add': []}
    for main_harm_boy_vedushego_vala in main_harm:
        freqs['Main'].append(freq * c_r_v * main_harm_boy_vedushego_vala - c_r_v)
        freqs['Main'].append(freq * c_r_v * main_harm_boy_vedushego_vala + c_r_v)
        freqs['Add'].append(freq * c_r_v * main_harm_boy_vedushego_vala - c_r_v)
        freqs['Add'].append(freq * c_r_v * main_harm_boy_vedushego_vala + c_r_v)
    return freqs


def fault_belt(freq, **param):
    """Дефект ремня

                Add some param

                Params:
                    freq: integer - Частота питающего напряжения.

                returns:
                    freqs (dict) {'Main: {Гармоника основной частоты: Список частот},

                                        'Add:{Гармоника основной частоты: Список частот}}.

                """
    diameter_first = param['Диаметр ведущего']
    diameter_second = param['Диаметр ведомого']
    len_belt = param['Длина ремня']
    c_v_r = freq_rot_vedomogo(freq, diameter_first, diameter_second)
    c_r_r = freq_rot_belt(freq, diameter_first, len_belt)
    main_harm_defekt_remnya = np.array([1, 2, 3, 4, 5])
    freqs = {'Main': [], 'Add': []}
    for harm in main_harm_defekt_remnya:
        freqs['Main'].append(harm * c_r_r)
        freqs['Main'].append(freq - harm * c_v_r)
        freqs['Main'].append(freq + harm * c_v_r)
        freqs['Main'].append(c_v_r - harm * c_v_r)
        freqs['Main'].append(c_v_r + harm * c_v_r)
        freqs['Add'].append(harm * c_r_r)
        freqs['Add'].append(freq - harm * c_v_r)
        freqs['Add'].append(freq + harm * c_v_r)
        freqs['Add'].append(c_v_r - harm * c_v_r)
        freqs['Add'].append(c_v_r + harm * c_v_r)

    return freqs


def add_fault(wave, freqs, coef_harm):
    for harm in freqs.keys():
        for one_freq, one_coef in zip(freqs[harm], coef_harm[harm]):
            print(f"Add Freq: {one_freq} with Coef: {one_coef}")
            signal = f5s.SinSignal(one_freq, amp=one_coef)
            fault_wave = signal.make_wave(wave.duration, wave.framerate)
            wave += fault_wave
    return wave

def create_coef(freqs, value):
    coef_dict = {}
    for harm in freqs:
        coef_dict[harm] = value * np.ones_like(freqs[harm])
    return coef_dict

def freqs_to_coefs_dict_ones(freqs):
    coef = freqs
    for harm in freqs:
        coef[harm] = [1] * len(freqs[harm])
    return coef

heuroistic_component = {
                    'Вал':[],
                    'Лопасти': [],
                    'Редуктор': [],
                    'Подшипники': [uneven_rad_loaded
                                  ]
                    }

# heuroistic_component = {
#                     'Вал':[],
#                     'Лопасти': [],
#                     'Редуктор': [],
#                     'Подшипники': [shaft_shaking, uneven_rad_loaded, swash_out_ring,
#                                      wear_out_ring, sink_out_ring, wear_inner_ring,
#                                      sink_inner_ring, weak_sep, break_inner_body,
#                                      defect_clutch
#                                   ]
#                     }
class Ope():
    def __init__(self, freq, operations, kwargs):
        self.operations = operations
        self.kwargs = kwargs
        self.freq = freq

    def __call__(self, *args, **kwargs):
        signal = f5s.SinSignal(0)
        for cor_opera in self.operations:
            pass


def ga(start, operation, search_space, real_signal):
    pass


operation = []
