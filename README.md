# localization-cin-es290
Localization project for the ES290 class (Mobile Communications) at CIn/UFPE


Regressor simples

	- Regressor1.py - utilizando os RSSI como entrada e a LAT/LON como saída

Com fingerprint (não precisa rodar fingerprint.py e fpMatchMedicao.py todas as vezes, pois demora muito):

	fingerprint.py - gerar grid
	fp-match-medicao.py - para cada ponto, achar a celula mais proxima

	fp-regressor.py - para cada ponto do grid, calcular os RSSI
	fp-knn.py - com os rssi de entrada calculados anteriormente, calcular as posicoes no grid
	fp-test.py - usando os rssi disponiveis, calcula a posicao final

	fp-classifier.py - roda um classificador em cima das celulas do fingerprint



Erros Médios:

	Regressor1.py (Regressor KNN): 145m
	fp-test.py (Fingerprint com regressor KNN): 252m
	fp-classifier.py (Fingerprint com classificador KNN): 180m
