import numpy as np
import itertools
import matplotlib.pyplot as plt

delta_ri = 10 ** -3  # comprimento delta_r da malha


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


def sol_continua_velocidade(r_inicial, r_e, vel_ang_i, vel_ang_e):
    delta_r = delta_ri
    n = int(1 + (r_e - r_inicial) / delta_r)

    def c1():
        return (vel_ang_e * r_e ** 2 - vel_ang_i * r_inicial ** 2) / (r_e ** 2 - r_inicial ** 2)

    def c2():
        return (r_inicial ** 2 * r_e ** 2 * (vel_ang_i - vel_ang_e)) / (r_e ** 2 - r_inicial ** 2)

    r = np.linspace(start=r_inicial, stop=r_e, num=n)
    v_teta_r = []

    c1 = c1()
    c2 = c2()

    for i in r:
        v_teta_r.append(c1 * i + c2 / i)

    return v_teta_r, r, delta_r


def plot_velocidades(v_teta_analitico, r_vetor_analitico, delta_r_analitico,
                     v_teta_numerico, r_vetor_numerico, delta_r_numerico,
                     nome_fluido, r_inicial, r_externo):
    plt.plot(r_vetor_analitico, v_teta_analitico)
    plt.plot(r_vetor_numerico, v_teta_numerico)
    plt.xlabel("r")
    plt.ylabel("V_teta(r)")
    plt.grid(True)
    plt.legend([f'Solução contínua, refinamento --> dr = {delta_r_analitico}',
                f'Solução computacional, refinamento --> dr = {delta_r_numerico}'])
    plt.title(f'Balanço da quantidade de movimento linear - Fluido: {nome_fluido}')

    # Ajustando o nome do arquivo para incluir o nome do fluido e os raios
    nome_arquivo = f'Velocidade_{nome_fluido}_Rinicial_{r_inicial}_Rexterno_{r_externo}_delta_r_{delta_r_numerico}.png'
    plt.savefig(nome_arquivo, format='png')
    plt.show()


def main():
    from reynolds_calculadora import calcular_reynolds

    df_resultados = calcular_reynolds()

    omega_externo = 100  # rad/s
    for index, row in df_resultados.iterrows():
        # Usando os valores do DataFrame nas funções
        v_teta_analitico, r_vetor_analitico, delta_r_analitico = sol_continua_velocidade(
            r_inicial=row['Raio Interno (m)'],
            r_e=row['Raio Externo (m)'],
            vel_ang_i=row['Velocidade Angular Interna Crítica ω_interno (rad/s)'],
            vel_ang_e=omega_externo
        )

        v_teta_numerico, r_vetor_numerico, delta_r_numerico = sol_numerica_velocidade(
            r_interno=row['Raio Interno (m)'],
            r_externo=row['Raio Externo (m)'],
            velocidade_angular_externa=omega_externo,
            velocidade_angular_interna=row['Velocidade Angular Interna Crítica ω_interno (rad/s)']
        )

        plot_velocidades(v_teta_analitico, r_vetor_analitico, delta_r_analitico,
                         v_teta_numerico, r_vetor_numerico, delta_r_numerico,
                         nome_fluido=row['Fluido'],
                         r_inicial=row['Raio Interno (m)'],
                         r_externo=row['Raio Externo (m)'])


if __name__ == "__main__":
    main()