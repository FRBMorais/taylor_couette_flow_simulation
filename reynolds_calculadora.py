import pandas as pd


def calcular_reynolds():
    # Configurações do pandas
    pd.set_option('display.max_columns', None)

    # Propriedades dos fluidos -- TODAS AS PROPRIEDADES FORAM EXTRAIDAS CONSIDERADAS CONSTANTES A 300K
    fluidos = {
        'Água': {
            'densidade': 995.6,  # kg/m^3 -- OK
            'viscosidade_dinamica': 0.0007972,  # Pa·s
            'condutividade_termica': 0.598,  # W/(m·K)
            'viscosidade_cinematica': 0.0000008007  # m²/s
        },
        'Óleo de motor': {  # propriedade retirada das tabelas do INCROPERA -- OK
            'densidade': 884.1,  # kg/m^3
            'viscosidade_dinamica': 0.486,  # Pa·s
            'condutividade_termica': 0.145,  # W/(m·K)
            'viscosidade_cinematica': 0.00055  # m²/s
        },
        'Glicerina': {  # propriedade retirada das tabelas do INCROPERA -- OK
            'densidade': 1259.9,  # kg/m^3
            'viscosidade_dinamica': 0.799,  # Pa·s
            'condutividade_termica': 0.286,  # W/(m·K)
            'viscosidade_cinematica': 0.000634  # m²/s
        },
        'Ar': {  # propriedade retirada das tabelas do INCROPERA -- OK
            'densidade': 1.1614,  # kg/m^3
            'viscosidade_dinamica': 0.00001846,  # Pa·s
            'condutividade_termica': 0.0263,  # W/(m·K)
            'viscosidade_cinematica': 0.00001589  # m²/s
        }
    }

    # Parâmetros dos cilindros -- valores aleatorios, podem ser alterados para os desejados
    raio_interno = 0.05  # metros
    omega_externo = 100  # rad/s

    # Gaps a serem considerados
    gaps = [0.005, 0.010, 0.015, 0.020]

    # Número de Reynolds crítico para a transição para turbulência
    Re_critico = 2300

    # Lista para guardar os resultados
    resultados = []

    # Cálculo da velocidade angular interna crítica
    for gap in gaps:
        raio_externo = raio_interno + gap
        for nome_fluido, propriedades in fluidos.items():
            densidade = propriedades['densidade']
            viscosidade_dinamica = propriedades['viscosidade_dinamica']
            viscosidade_cinematica = propriedades['viscosidade_cinematica']
            condutividade_termica = propriedades['condutividade_termica']
            omega_interno_critico = (Re_critico * viscosidade_dinamica) / (densidade * gap * raio_interno) + (omega_externo * raio_externo / raio_interno)
            resultados.append({
                'Raio Interno (m)': raio_interno,
                'Raio Externo (m)': raio_externo,
                'Gap (m)': gap,
                'Velocidade Angular Interna Crítica ω_interno (rad/s)': omega_interno_critico,
                'Fluido': nome_fluido,
                'Densidade (kg/m^3)': densidade,
                'Viscosidade dinamica (Pa.s)': viscosidade_dinamica,
                'Viscosidade cinematica (m²/s)': viscosidade_cinematica,
                'Condutividade Termica (W/(m·K))': condutividade_termica
            })

    # Convertendo os resultados para um DataFrame
    df_resultados = pd.DataFrame(resultados)

    return df_resultados
