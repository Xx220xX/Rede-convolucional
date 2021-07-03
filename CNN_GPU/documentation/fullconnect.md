<!DOCTYPE html>
<html>
<head>
	<title>Camada FullConnect</title>
</head>
<body>
	<section>
		<h2>Teoria</h2>
		<p>
			Esta camada possui grande importância no aprendizado da rede, 
			uma vez que os neuronios de saída estão  todos conectados aos neuronios de entrada.
			Independente da dimensão da entrada, a dimensão de saída é definida sem restrição.
		</p>
		<h3>Predição</h3>
		<p>
			Também  chamado  de feedforward, é o processo que gera uma saída
			a partir de uma entrada. Consiste na multiplicação matricial dos seus pesos pela entrada e em seguida aplicando uma
			<a href="funcaoAtivação.md">função de ativação<a/>.
			</p>
			<p paddingLeft=1%>
				<img src="https://latex.codecogs.com/svg.latex?\Large&space;a" alt="matriz de entrada" width="2%" align="left">
				Matriz de entrada ordem nx1x1
			</p>
			<p paddingLeft=1%>
				<img src="https://latex.codecogs.com/svg.latex?\Large&space;w" alt="matriz peso" width="2%" align="left">
				Matriz de pesos ordem mx1x1
			</p>
			<p>É feito a multiplicação de matrizes</p>
			<p>
				<img src="https://latex.codecogs.com/svg.latex?\Large&space;z=w\times%20a" alt="matriz z" width="12%" align="left">
				resulta em uma matriz mx1x1.
			</p>
			<p>
				Estudos mostraram que uma rede neural com apenas uma multiplicação linear é capaz de resolver apenas problemas lineares.
				Portanto é aplicado uma função não linear, agora podendo resolver problemas não lineares.
			</p>
			<p>
				<img src="https://latex.codecogs.com/svg.latex?\Large&space;s=f(z)" alt="matriz s" width="10%" align="left">
				resulta em uma matriz mx1x1 a qual é a saída desta camada.
			</p>
			<h3>Retropropagação</h3>
			<p>Para esta  camada o método  utilizado é o gradiente descendente. Pela teoria do cálculo 
				o gradiente fornece a direção  de crescimento de uma função, para mostrar isso basta analisar a definição 
				da derivada. (Por simplicidade considerar espaço R²).
			</p>
			<h4>o que é Derivada e qual sua importância para o método gradiente descendente</h4>
			<div style = "padding-left:40px">
				<p>Seja <img src="https://latex.codecogs.com/svg.latex?\Large&space;y=g(x)" alt="y igual a g de x" width="8%" align="">
					uma função contida em R².
					<p>
						A derivada é definida por:
					</p>
					<p>
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;y'=f'(z)=\frac{\mathrm{d}g(x)}{\mathrm{d}x}=\lim_{h\to0}\frac{g(x+h)-g(x)}{g}" alt="definição derivada" width="40%" align="above">
					</p>
				</p>
				<div>
					<p>
						<img style="float:left;padding-right:10px;padding-bottom:10px;" src="img/derivada.jpg" alt="derivada grafico" width="30%" >
						Ao lado é visto graficamente. O  subtração dos pontos
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x+h)" alt="" width="8%" align="">
						e
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x)" alt="" width="4%" align="">
						fornece o cateto oposto do triangulo formado.
						E o <img src="https://latex.codecogs.com/svg.latex?\Large&space;h" alt="" width="1%" align=""> o cateto adjacente.
					</p>
				</div>
				<div style="clear:left;">
					<p  >
						<img style="float:left;padding-right:10px;padding-bottom:10px;" src="img/derivada2.jpg" alt="derivada grafico2" width="30%" align="">
						Conforme mostrado na figura, a derivada então equivale ao calculo da tangente no ponto, uma vez que o limite está aplicado.
						Se o ponto 
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x+h)" alt="" width="8%" align="">
						estiver mais alto que o ponto
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x)" alt="" width="4%" align="">
						então a inclinação é positiva logo a tangente é positiva e a derivada também positiva, indicando que a medida que o valor
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;x" alt="" width="2%" align="">
						aumenta (vai para a direita) a função 
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x)" alt="" width="4%" align="">
						tende a aumentar.
						<p>
							Caso o ponto 
							<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x+h)" alt="" width="8%" align="">
							estiver mais baixo que o ponto
							<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x)" alt="" width="4%" align="">
							então a inclinação é negativa logo a tangente é negativa e a derivada também negativa, indicando que a medida que o valor
							<img src="https://latex.codecogs.com/svg.latex?\Large&space;x" alt="" width="2%" align="">
							diminui (vai para a esquerda) a função 
							<img src="https://latex.codecogs.com/svg.latex?\Large&space;g(x)" alt="" width="4%" align="">
							tende a aumentar.
						</p>
					</p>
				</div>
				<div style="clear:left;">
					<p>
						Resumindo, a derivada fornece a direção de crescimento da função. Para o caso de 
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;R^n" alt="" width="2%" align="">
						é chamado de <a href="https://www.infopedia.pt/dicionarios/lingua-portuguesa/gradiente">Gradiente</a>
					</p>
				</div>
			</div>
			<h4>Aplicando o método gradiante descendente</h4>
			<div style="padding-left:40px">
				<p>
					Sabendo que é possível determinar a direção de crescimento de uma função, o próximo passo é definir uma função 
					para ser maximizada (no caso uma função de recompensa) ou uma função a ser minimizada (uma função de erro).
				</p>
				<p>
					No gradiente descendente é usado a função de erro médio quadrático, o objetivo é diminuir o erro final.
					A preocupação agora é em determinar o gradiente da matriz 
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;w" alt="" width="2%" align="">
					para então caminhar na direção oposta a fim de diminuir o erro. (Caminhar na direção do gradiente vai fazer com que o erro aumente).				
				</p>
				<p>
					Considerando a função de erro como
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;E(w)" alt="" width="4%" align="">
					queremos encontrar 
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial%20E(w)}{\partial%20w}" alt="" width="4%" align="">
					.
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial%20E(w)}{\partial%20w}=\frac{\partial%20z}{\partial%20w}\cdot\frac{\partial%20E}{\partial%20z}" alt="" height="40px" align="">
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial%20E(w)}{\partial%20w}=\frac{\partial%20z}{\partial%20w}\cdot\frac{\partial%20s}{\partial%20z}\cdot\frac{\partial%20E}{\partial%20s}" alt="" height="40px" align="">
				</p>
				<p>
					Fazendo as derivadas obtém-se:
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;ds=\frac{\partial%20E}{\partial%20s}" alt="" height="40px" align="">
					Este é fornecido pelas próximas camadas, caso seja a ultima camada, então :
					<p style="padding-left:50px">
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;ds= s - t" alt="" height="18px" align="">, onde 
						<img src="https://latex.codecogs.com/svg.latex?\Large&space;t" alt="" width="1%" align=""> é o vetor de saída esperada no final da rede.
					</p>
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial%20s}{\partial%20z}=\frac{\mathrm{d}f(z)}{\mathrm{d}z}" alt="" height="40px" align="">
					Este é a derivada da função de ativação.
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;\frac{\partial%20z}{\partial%20w}=a" alt="" height="40px" align="">
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;dz=\frac{\partial%20E}{\partial%20z}=\frac{\mathrm{d}f(z)}{\mathrm{d}z}\cdot%20ds" alt="" height="40px" align="">
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;dw=\frac{\partial%20E}{\partial%20w}=dz\times%20a^T" alt="" height="40px" align="">
				</p>
				<p>
					Com o gradiente dos pesos definidos os pesos são corrigidos andando na direção oposta do gradiente
					<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;w_{t+1}=w_{t}-h\cdot%20dw" alt="" height="20px" align="">
					, Onde h é um parâmetro chamado taxa de aprendizado. Usado pois o oposto do gradiente indica apenas a direção que a função decresce, não informando a distância até o minimo.
					</p>
				</p>
				<p>
					O gradiente de entrada, útil para as camadas anteriores aplicarem este método, é expresso por:
				</p>
				<img src="https://latex.codecogs.com/svg.latex?\Large&space;da=\frac{\partial%20E}{\partial%20a}=\frac{\partial%20z}{\partial%20a}\cdot\frac{\partial%20E}{\partial%20z}=w^T\times%20dz" alt="" height="40px" align="">
				</p>
			</div>
		</section>
		<section>
			<h2>Exemplo numérico</h2>
			<div>
				<p>
					Seja uma camada fullconnect com entrada 3x1x1 e saída 2x1x1 com taxa de aprendizado igual a 1 . A matriz w aleatoriamente inicializada:
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;w=\begin{vmatrix}0.3&-0.2&-0.7\\0.5&0.4&-0.1\end{vmatrix}" alt="" height="80px" align="">
				</p>
				<p>
					Para uma entrada "a", a saída esperada é "t".
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;a=\begin{vmatrix}0.1\\0.4\\-0.6\end{vmatrix}" alt="" height="100px" align="">
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;t=\begin{vmatrix}0.8\\0.5\end{vmatrix}" alt="" height="80px" align="">
				</p>
				<p>
					A primeira parte é a obtenção da matriz z, resultante da multiplicação de "w" por "a"
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;z=w\times%20a=\begin{vmatrix}0.37\\0.27\end{vmatrix}" alt="" height="80px" align="">
				</p>
				<p>
					Passando pela função de ativação (para este exemplo a sigmoid).
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;s=f(z)=\begin{vmatrix}0.591\\0.567\end{vmatrix}" alt="" height="80px" align="">
				</p>
				<p>
					A saída obtida é diferente da esperada, então é feito a retropropagação a fim de obter uma resposta mais próxima. 
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;ds=s-t=\begin{vmatrix}-0.209\\0.067\end{vmatrix}" alt="" height="80px" align="">
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;dz=f'(z)\cdot%20(s-t)=\begin{vmatrix}-0.050\\0.016\end{vmatrix}" alt="" height="80px" align="">
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;dw=dz\times%20a^T=\begin{vmatrix}-0.005&-0.02&0.03\\0.002&0.007&-0.01\end{vmatrix}" alt="" height="80px" align="">
				</p>
				<p>
					Assim os pesos podem ser corrigidos para :
				</p>
				<p>
					<img src="https://latex.codecogs.com/svg.latex?\Large&space;w=w_{t-1}-dw_{t-1}=\begin{vmatrix}0.305&-0.180&-0.730\\0.498&0.393&-0.090\end{vmatrix}" alt="" height="80px" align="">
				</p>
				<p>
					Este processo é repetido  diversas vezes até atingir um erro final permitido dependendo da aplicação.
				</p>
			</div>
		</section>
	</body>
	</html>
