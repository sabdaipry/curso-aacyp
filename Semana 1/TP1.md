# Trabajo Práctico N° 1: Introducción a Análisis Estadístico

## Consignas

1. Del libro “Bishop, C. Pattern Recognition and machine learning” resolver los ejercicios 1.2, 1.3, 1.5, 1.6, 1.11,1.21, y 1.30.
2. Para el caso de 1.30, generar simulaciones de la divergencia KL cuando la primera distribución está normalizada. Luego repetir intercambiando por la segunda. Generar visualización de KL respecto de las varianzas y respecto de las medias.

---

## Ejercicio 1.2

### Enunciado
Escriba el conjunto de ecuaciones lineales acopladas, análogas a (1.122), que satisfacen los coeficientes $w_i$ que minimizan la función de error de suma de cuadrados regularizada dada por (1.4).

$\sum_{j=0}^N A_{ij}w_j=T_i$      (1.122)

donde:

$A_{ij}=\sum_{n=1}^N (x_n)^{i+j}$ y $T_i=\sum_{n=1}^N (x_n)^it_n$ (1.123)

$\tilde{E}(w)=\frac{1}{2}\sum_{n=1}^N \{y(x_n, w)-t_n\}^2 + \frac{\lambda}{2}||w||^2$ (1.4)

donde $||w||^2 \equiv w^Tw=w_0^2+w_1^2 + ...+w_M^2$ y el coeficiente $\lambda$ regula la importancia relativa del término de regularización en comparación con el término de error de suma de cuadrados.

### Desarrollo

El objetivo es minimizar la función de error regularizada $\tilde{E}(w)$:

$$\tilde{E}(w)=\frac{1}{2}\sum_{n=1}^N \{y(x_n, w)-t_n\}^2 + \frac{\lambda}{2}||w||^2$$

Para encontrar el mínimo, se debe calcular la derivada parcial de $\tilde{E}(w)$ con respecto a cada coeficiente $w_i$ (para $i=0,1,….,M$) e igualar el resultado a cero.

Primero, se sustituye la definición del polinomio $y(x_n, w)=\sum_{j=0}^M w_jx_n^j$ y la norma al cuadrado $||w||^2=\sum_{j=0}^M w_j^2$ en la ecuación del error:

$$\tilde{E}(w)=\frac{1}{2}\sum_{n=1}^N \left(\sum_{j=0}^M w_jx_n^j-t_n\right)^2 + \frac{\lambda}{2} \sum_{j=0}^M w_j^2$$

Ahora, se procede a derivar esta expresión con respecto a un coeficiente específico w_i e igualar a cero.

$$\frac{\partial\tilde{E}(w)}{\partial w_i}=\frac{\partial}{\partial w_i}\left[\frac{1}{2}\sum_{n=1}^N \left(\sum_{j=0}^M w_jx_n^j-t_n\right)^2\right] + \frac{\partial}{\partial w_i}\left[\frac{\lambda}{2} \sum_{j=0}^M w_j^2\right]=0$$

Se procede a calcular la derivada de cada término por separado:

1. Derivada del término de suma de cuadrados:
    Usando la regla de la cadena, la derivada de este término es idéntica a la del ejercicio no regularizado. La derivada de la expresión interna ($\sum_{j=0}^M w_jx_n^j-t_n$) con respecto a $w_i$ es simplemente $x_n^i$.

   $$\frac{\partial}{\partial w_i}\left[\frac{1}{2}\sum_{n=1}^N \left(\sum_{j=0}^M w_jx_n^j-t_n\right)^2\right]=\frac{1}{2}\sum_{n=1}^N 2\left(\sum_{j=0}^M w_jx_n^j-t_n\right)x_n^i$$

2. Derivada del término de regularización: La derivada de $\sum_{j=0}^M w_j^2$ con respecto a $w_i$ es $2 w_i$ (todos los demás términos $w_j^2$ son constantes respecto a $w_i$).

   $$\frac{\partial}{\partial w_i}\left[\frac{\lambda}{2} \sum_{j=0}^M w_j^2\right]=\frac{\lambda}{2}(2w_i)=\lambda w_i$$

Ahora, se unen ambos resultados y se iguala a cero:

$$\sum_{n=1}^N \left(\sum_{j=0}^M w_jx_n^j-t_n\right)x_n^i+\lambda w_i=0$$

Para llegar a la forma deseada (análoga a 1.122), se reorganizan los términos. Primero, se distribuye $x_n^i$:

$$\sum_{n=1}^N \left(\sum_{j=0}^M w_jx_n^jx_n^i\right)-\sum_{n=1}^N t_n x_n^i + \lambda w_i=0$$

Se combinan las potencias de $x_n$ y se mueve el término con $t_n$, hacia la derecha:

$$\sum_{n=1}^N\sum_{j=0}^M w_jx_n^{i+j} +\lambda w_i=\sum_{n=1}^N t_n x_n^i$$

Se intercambia el orden de las sumatorias en el primer término y se saca $w_j$ fuera de la suma interna (ya que no depende de $n$):

$$\sum_{j=0}^Mw_j \left(\sum_{n=1}^N x_n^{i+j}\right) + \lambda w_i=\sum_{n=1}^N t_n x_n^i$$

Se reconocen los términos $A_{ij}=\sum_{n=1}^N (x_n)^{i+j}$ y $T_i=\sum_{n=1}^N (x_n)^it_n$ de 1.123:

$$\sum_{j=0}^M A_{ij} w_j + \lambda w_i=T_i$$

Para incorporar el término $\lambda w_i$ dentro de la sumatoria, se usa la Delta de Kronecker ($\delta_{ij}$), que vale 1 si $i=j$ y 0 en caso contrario. Así, $\lambda w_i=\sum_{j=0}^M \lambda \delta_{ij} w_j$.

$$\sum_{j=0}^M A_{ij} w_j + \sum_{j=0}^M \lambda \delta_{ij} w_j = T_i$$

Finalmente, se combinan las sumatorias:

$$\sum_{j=0}^M \left(A_{ij}+\lambda \delta_{ij}\right)w_j=T_i$$

Entonces, el conjunto de ecuaciones lineales que minimiza la función de error regularizada es:

$$\sum_{j=0}^M \tilde{A}_{ij}w_j=T_i$$

Donde:

$$\tilde{A}_{ij}=\sum_{n=1}^N(x_n)^{i+j}+\lambda\delta_{ij}$$
$$T_i=\sum_{n=1}^N(x_n)^i t_n$$

La única diferencia con el caso no regularizado (Ejercicio 1.1) es que al término $A_{ij}$ se le suma el término de regularización λ únicamente en la diagonal de la matriz (cuando $i=j$), debido a la Delta de Kronecker. Este método se conoce como Regresión de Ridge.

---

## Ejercicio 1.3

### Enunciado

Supongamos que tenemos tres cajas de colores r (roja), b (azul) y g (verde). La caja r contiene 3 manzanas, 4 naranjas y 3 limas. La caja b contiene 1 manzana, 1 naranja y 0 limas. La caja g contiene 3 manzanas, 3 naranjas y 4 limas. Si se elige una caja al azar con probabilidades $p(r)=0,2$, $p(b)=0,2$, $p(g)=0,6$, y se saca una fruta de la caja (con igual probabilidad de seleccionar cualquiera de los artículos de la caja), entonces:

1. ¿Cuál es la probabilidad de seleccionar una manzana?
2.	Si observamos que la fruta seleccionada es una naranja, ¿cuál es la probabilidad de que provenga de la caja verde?

### Desarrollo

Primeramente, se organiza la información. Se utilizará la letra ‘a’ para manzana, ‘o’ para naranja y ‘l’ para lima.

**Probabilidades a priori de elegir una caja:**
* $P(caja=r)=0,2$
* $P(caja=b)=0,2$
* $P(caja=g)=0,6$

**Composición de las cajas y probabilidades condicionales de sacar una fruta dada una caja:**

**Caja Roja (r)**

$3a+4o+3l=10 frutas$

* $P(fruta=a│caja=r)=\frac{3}{10}=0,3$
* $P(fruta=o│caja=r)=\frac{4}{10}=0,4$
* $P(fruta=l│caja=r)=\frac{3}{10}=0,3$

**Caja Azul (b)**

$1a+1o+0l=2frutas$

* $P(fruta=a│caja=b)=\frac{1}{2}=0,5$
* $P(fruta=o│caja=b)=\frac{1}{2}=0,5$

**Caja Verde (g)**

$3a+3o+4l=10frutas$

* $P(fruta=a│caja=g)=\frac{3}{10}=0,3$
* $P(fruta=o│caja=g)=\frac{3}{10}=0,3$
* $P(fruta=l│caja=g)=\frac{4}{10}=0,4$

#### Punto 1

Para encontrar la probabilidad total de seleccionar una manzana, $P(a)$, se usa la Ley de Probabilidad Total. Se suma la probabilidad de sacar una manzana de cada caja, ponderada por la probabilidad de haber elegido esa caja.

$$P(a)=P(a │ r)P(r)+P(a │ b)P(b)+P(a │ g)P(g)$$

Sustituyendo con los valores calculados:

$$P(a)=(0,3\times0,2)+(0,5\times0,2)+(0,3\times0,6)$$

$$P(a)=0,06+0,10+0,18$$

$$P(a)=0,34$$

La probabilidad de seleccionar una manzana es de 0.34 o del 34 %.

#### Punto 2

En este caso, se trata de una probabilidad condicional: “sabiendo que la fruta es una naranja, ¿cuál es la probabilidad de que la caja sea verde?”. Esto se escribe como $P(caja=g|fruta=o)$ y es un caso clásico para aplicar el Teorema de Bayes.

La fórmula del Teorema de Bayes es:

$$P(g|o)=\frac{P(o|g)P(g)}{P(o)}$$

Los términos del numerador ya son conocidos:

* $P(o│g)=0,3$ (probabilidad de sacar una naranja de la caja verde)
* $P(g)=0,6$ (probabilidad de elegir la caja verde)

La probabilidad total de sacar una naranja $P(o)$ se calcula igual que en ítem 1, usando la Ley de Probabilidad Total:

$$P(o)=P(o∣r)P(r)+P(o∣b)P(b)+P(o∣g)P(g)$$

$$P(o)=(0,4\times0,2)+(0,5\times0,2)+(0,3\times0,6)$$

$$P(o)=0,08+0,10+0,18$$

$$P(o)=0,36$$

Ahora se aplica Bayes:

$$P(g|o)=\frac{0,3\times0,6}{0,36}=0,5$$

Si la fruta seleccionada es una naranja, la probabilidad de que provenga de la caja verde es de 0,5 o 50 %.

---

## Ejercicio 1.5












