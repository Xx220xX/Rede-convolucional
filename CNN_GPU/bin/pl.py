import csv
import matplotlib.pyplot as plt
with open('onlygpuMulttime.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	lines = []
	for row in csv_reader:
		lines.append(row)
	title = lines.pop(0)
	print(title)
	m = [float(line.pop(0))  for line in lines]
	onlyHost = [float(line.pop(0)) for line in lines]
	host_svm = [float(line.pop(0)) for line in lines]
	svm_gpu = [float(line.pop(0)) for line in lines]
	gpu = [float(line.pop(0)) for line in lines]


# plt.plot(m,onlyHost)
# plt.plot(m,host_svm)
plt.plot(m,svm_gpu)
plt.plot(m,gpu)
plt.legend(title[3:])
plt.xlabel('Dimensão da matriz quadrada')
plt.ylabel('milissegundos')
plt.title('Comparativo de performance(Tempo para multiplicação)')
# plt.xticks(list(range(1,m[-1],10)))
plt.show()