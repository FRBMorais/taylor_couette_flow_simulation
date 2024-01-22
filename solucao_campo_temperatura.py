import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

delta_ri = 10 ** -5  # comprimento delta_r da malha


def sol_numerica_velocidade(r_interno, r_externo, velocidade_angular_externa, velocidade_angular_interna):
    delta_r = delta_ri  # delta_r da discretizacao do dominio
    n_pontos = int(1 + (r_externo - r_interno) / delta_r)  # quantidade de pontos discretizados do problema

    r = np.linspace(start=r_interno, stop=r_externo, num=n_pontos)  # [m] --> domínio do problema
    r_i_menos_meio = [r_i - (delta_r / 2) for r_i in r]
    r_i_mais_meio = [r_i + (delta_r / 2) for r_i in r]

    def d_r(raio):
        return 1 / raio

    x = range(2, n_pontos)

    vetor_a = [(- 2 / delta_r ** 2)
               - (d_r(r_i_mais_meio[i]) / (2 * delta_r))
               + (d_r(r_i_menos_meio[i]) / (2 * delta_r))
               - (1 / r[i] ** 2) for i in x]

    vetor_b = [((1 / delta_r ** 2)
                + (d_r(r_i_mais_meio[i]) / (2 * delta_r))) for i in x]

    vetor_c = [((1 / delta_r ** 2)
                - (d_r(r_i_menos_meio[i]) / (2 * delta_r))) for i in x]

    v1 = r_interno * velocidade_angular_interna
    vn = r_externo * velocidade_angular_externa
    linha_1 = np.zeros(n_pontos)
    linha_n = np.zeros(n_pontos)
    funcao_resposta = np.zeros(n_pontos)

    linha_1[0] = 1
    linha_n[n_pontos - 1] = 1
    funcao_resposta[0] = v1
    funcao_resposta[n_pontos - 1] = vn

    linhas_internas = np.zeros(n_pontos * (n_pontos - 2))

    indices = []

    contador = 1
    for i in range(len(linhas_internas)):
        contador -= 1
        if contador == 0:
            for j in range(1):
                indices.append(i)
            contador = n_pontos + 1

    contadora = 0
    for i in range(len(linhas_internas)):
        if i in indices:
            linhas_internas[i] = vetor_c[contadora]
            linhas_internas[i + 1] = vetor_a[contadora]
            linhas_internas[i + 2] = vetor_b[contadora]
            contadora += 1

    matriz_coeficientes = list(itertools.chain(linha_1, linhas_internas, linha_n))
    v = np.array(matriz_coeficientes).reshape(n_pontos, n_pontos)
    inversa = np.linalg.inv(v)

    respostaresposta = np.array(funcao_resposta).reshape(n_pontos, 1)

    v_resposta = np.dot(inversa, respostaresposta)

    v_resposta.transpose()
    return v_resposta, r, delta_r


t2 = 30  # temperatura inicial do cilindro externo


def sol_continua_temperatura(r_inicial,r_eterno, vel_ang_i, vel_ang_e, mi, k):
    delta_r = delta_ri
    n = int(1 + (r_eterno - r_inicial) / delta_r)

    def c2():
        return (r_inicial ** 2 * r_eterno ** 2 * (vel_ang_i - vel_ang_e)) / (r_eterno ** 2 - r_inicial ** 2)

    def ct1():
        return (-2 * mi * c2() ** 2) / (k * r_inicial ** 2)

    def ct2():
        return t2 + (mi * c2() ** 2 / k) * ((1 / r_eterno ** 2) + (2 * np.log(r_eterno) / r_inicial ** 2))

    r = np.linspace(start=r_inicial, stop=r_eterno, num=n)
    t_r = []
    ct11 = ct1()
    ct22 = ct2()
    c22 = c2()

    for i in r:
        t_r.append((-c22 ** 2 * mi / k) * (1 / i ** 2) + ct11 * np.log(i) + ct22)

    return t_r, r, delta_r


def sol_numerica_temperatura(r_interno, r_externo, v_teta_numerico, mi, k):
    delta_r = delta_ri  # delta_r da discretizacao do dominio

    n_pontos = int(1 + (r_externo - r_interno) / delta_r)  # quantidade de pontos discretizados do problema

    r = np.linspace(start=r_interno, stop=r_externo, num=n_pontos)  # [m] --> domínio do problema
    r_i_menos_meio = [r_i - (delta_r / 2) for r_i in r]
    r_i_mais_meio = [r_i + (delta_r / 2) for r_i in r]

    def d_r(raio):
        return 1 / raio

    x = range(2, n_pontos)

    vetor_a = [(- 2 / delta_r ** 2)
               - (d_r(r_i_mais_meio[i]) / (2 * delta_r))
               + (d_r(r_i_menos_meio[i]) / (2 * delta_r)) for i in x]

    vetor_b = [((1 / delta_r ** 2)
                + (d_r(r_i_mais_meio[i]) / (2 * delta_r))) for i in x]

    vetor_c = [((1 / delta_r ** 2)
                - (d_r(r_i_menos_meio[i]) / (2 * delta_r))) for i in x]

    quarto_termo = [-v_teta_numerico[i] / r[i] for i in range(1, n_pontos - 1)]
    terceiro_termo = [(v_teta_numerico[i + 1] - v_teta_numerico[i - 1])
                      / (2 * delta_r) for i in range(1, n_pontos - 1)]

    f_i = [(mi / k) * (terceiro_termo[i] + quarto_termo[i]) ** 2 for i in range(len(terceiro_termo))]
    linha_1 = np.zeros(n_pontos)
    linha_n = np.zeros(n_pontos)

    funcao_resposta = np.zeros(n_pontos)
    funcao_resposta[-1] = t2  # temperatura interna do cilindro externo
    for i in range(1, n_pontos - 1):
        funcao_resposta[i] = - f_i[i - 1][0]
    funcao_resposta[0] = 0

    linha_1[0] = 1
    linha_1[1] = ((2 * r_interno ** 2 - 2 * r_interno - delta_r ** 2) / (3 * delta_r ** 2)) - 1
    linha_1[2] = -((2 * r_interno ** 2 - 2 * r_interno - delta_r ** 2) / (3 * delta_r ** 2))
    linha_n[n_pontos - 1] = 1

    linhas_internas = np.zeros(n_pontos * (n_pontos - 2))

    indices = []

    contador = 1
    for i in range(len(linhas_internas)):
        contador -= 1
        if contador == 0:
            for j in range(1):
                indices.append(i)
            contador = n_pontos + 1

    contadora = 0
    for i in range(len(linhas_internas)):
        if i in indices:
            linhas_internas[i] = vetor_c[contadora]
            linhas_internas[i + 1] = vetor_a[contadora]
            linhas_internas[i + 2] = vetor_b[contadora]
            contadora += 1

    matriz_coeficientes = list(itertools.chain(linha_1, linhas_internas, linha_n))
    v = np.array(matriz_coeficientes).reshape(n_pontos, n_pontos)
    inversa = np.linalg.inv(v)

    respostaresposta = np.array(funcao_resposta).reshape(n_pontos, 1)
    t_resposta = np.dot(inversa, respostaresposta)

    t_resposta.transpose()
    return t_resposta, r, delta_r


def plot_temperaturas(t_r_analitico,
                      r_vetor_temp_analitico,
                      delta_r_temp_analitico,
                      t_r_numerico,
                      r_vetor_temp_numerico,
                      delta_r_temp_numerico,
                      nome_fluido,
                      r_interno,
                      r_externo):
    plt.plot(r_vetor_temp_analitico, t_r_analitico)
    plt.plot(r_vetor_temp_numerico, t_r_numerico)
    plt.xlabel("r")
    plt.ylabel("T(r)")
    plt.grid(True)
    plt.legend([f'Solucao contínua, refinamento --> dr = {delta_r_temp_analitico}',
                f'Solucao computacional, refinamento --> dr = {delta_r_temp_numerico}'])
    plt.title(f'Balanço energético - Fluido: {nome_fluido}')

    # Ajustando os eixos para não usar notação científica
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='both')

    # Formatando os rótulos para duas casas decimais
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.4f'))

    # Ajustando o nome do arquivo para incluir o nome do fluido e os raios
    nome_arquivo = f'Temperatura_{nome_fluido}_Rinicial_{r_interno}_Rexterno_{r_externo}_delta_r_{delta_r_temp_analitico}.png'
    plt.savefig(nome_arquivo, format='png')
    plt.show()


def main():
    from reynolds_calculadora import calcular_reynolds

    df_resultados = calcular_reynolds()

    omega_externo = 100  # rad/s
    for index, row in df_resultados.iterrows():
        # Usando os valores do DataFrame nas funções
        v_teta_numerico, r_vetor_numerico, delta_r_numerico = sol_numerica_velocidade(r_interno=row['Raio Interno (m)'],
                                                                                      r_externo=row['Raio Externo (m)'],
                                                                                      velocidade_angular_externa=omega_externo,
                                                                                      velocidade_angular_interna=row['Velocidade Angular Interna Crítica ω_interno (rad/s)'],
                                                                                      )
        t_r_analitico, r_vetor_temp_analitico, delta_r_temp_analitico = sol_continua_temperatura(r_inicial=row['Raio Interno (m)'],
                                                                                                 r_eterno=row['Raio Externo (m)'],
                                                                                                 vel_ang_i=row['Velocidade Angular Interna Crítica ω_interno (rad/s)'],
                                                                                                 vel_ang_e=omega_externo,
                                                                                                 mi=row['Viscosidade cinematica (m²/s)'],
                                                                                                 k=row['Condutividade Termica (W/(m·K))'])
        t_r_numerico, r_vetor_temp_numerico, delta_r_temp_numerico = sol_numerica_temperatura(r_interno=row['Raio Interno (m)'],
                                                                                              r_externo=row['Raio Externo (m)'],
                                                                                              v_teta_numerico=v_teta_numerico,
                                                                                              mi=row['Viscosidade cinematica (m²/s)'],
                                                                                              k=row['Condutividade Termica (W/(m·K))'])
        plot_temperaturas(t_r_analitico=t_r_analitico,
                          r_vetor_temp_analitico=r_vetor_temp_analitico,
                          delta_r_temp_analitico=delta_r_temp_analitico,
                          t_r_numerico=t_r_numerico,
                          r_vetor_temp_numerico=r_vetor_temp_numerico,
                          delta_r_temp_numerico=delta_r_temp_numerico,
                          r_interno=row['Raio Interno (m)'],
                          r_externo=row['Raio Externo (m)'],
                          nome_fluido=row['Fluido'])


if __name__ == "__main__":
    main()
