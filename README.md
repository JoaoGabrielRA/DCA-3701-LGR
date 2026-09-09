---
title: LGR - Lugar Geometrico das Raizes
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false

---

# LGR - Lugar Geometrico das Raizes

App em Streamlit para construir o Lugar Geometrico das Raizes (LGR) passo a passo
(DCA-3701 - Projeto de Sistemas de Controle - UFRN), incluindo:

1. Equacao caracteristica
2. Forma fatorada
3. Polos e zeros
4. Segmentos no eixo real
5. Numero de lugares separados
6. Simetria
7. Assintotas
8. Pontos de breakaway/break-in
9. Cruzamento com o eixo imaginario (Routh-Hurwitz e substituicao s = jw)
10. Angulos de partida/chegada
11. Criterio de angulo
12. Calculo de K

## Rodando localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```
