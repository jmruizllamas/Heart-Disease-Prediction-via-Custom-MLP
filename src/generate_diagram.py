import matplotlib.pyplot as plt
import numpy as np


def draw_neural_network(ax, left, right, bottom, top, layer_sizes, layer_labels):
    """
    Dibuja una red neuronal con estilo académico.
    """
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Colores para cada capa (Input: Azul, Hidden: Verde, Output: Rojo)
    colors = ['#4B8BBE', '#306998', '#FF6F61']

    # Guardar coordenadas de los nodos para dibujar las líneas
    layer_nodes = []

    # 1. Calcular posiciones de los nodos
    for i, n in enumerate(layer_sizes):
        layer_top = v_spacing * (n - 1) / 2. + (top + bottom) / 2.

        nodes_in_this_layer = []
        for m in range(n):
            x = left + i * h_spacing
            y = layer_top - m * v_spacing
            nodes_in_this_layer.append((x, y))

        layer_nodes.append(nodes_in_this_layer)

    # 2. Dibujar las conexiones (Líneas)
    # Las dibujamos primero para que queden DETRÁS de los círculos
    for i in range(len(layer_nodes) - 1):
        for node_a in layer_nodes[i]:
            for node_b in layer_nodes[i + 1]:
                # Alpha bajo para que las líneas no saturen el dibujo
                line_alpha = 0.2 if layer_sizes[i] > 10 else 0.5
                line = plt.Line2D([node_a[0], node_b[0]],
                                  [node_a[1], node_b[1]],
                                  c='gray', alpha=line_alpha, linewidth=0.5)
                ax.add_artist(line)

    # 3. Dibujar los Nodos (Círculos) y Etiquetas
    for i, nodes in enumerate(layer_nodes):
        color = colors[i % len(colors)]

        # Etiqueta de la capa (arriba)
        plt.text(nodes[0][0], top + 0.05, layer_labels[i],
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Info extra (función de activación)
        if i == 1:  # Hidden
            plt.text(nodes[-1][0], bottom - 0.05, "Activation:\nReLU",
                     ha='center', va='top', fontsize=10, style='italic', color='#444444')
        elif i == 2:  # Output
            plt.text(nodes[-1][0], bottom - 0.05, "Activation:\nSigmoid",
                     ha='center', va='top', fontsize=10, style='italic', color='#444444')

        for node in nodes:
            circle = plt.Circle(node, radius=0.025, color=color, zorder=4, ec='k')
            ax.add_artist(circle)


def main():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    ax.axis('off')

    # --- CONFIGURACIÓN DE TU ARQUITECTURA ---
    # 20 Entradas, 16 Ocultas, 1 Salida
    layer_sizes = [20, 16, 1]

    # Etiquetas para el gráfico
    labels = [
        "Input Layer\n(20 Features)",
        "Hidden Layer\n(16 Neurons)",
        "Output Layer\n(Probability)"
    ]

    draw_neural_network(ax, .1, .9, .1, .9, layer_sizes, labels)

    # Título del gráfico
    plt.title("Architecture of the Implemented MLP\n[20 - 16 - 1]", fontsize=14)

    # Guardar con alta calidad (DPI 300)
    output_filename = 'mlp_architecture_diagram.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico generado exitosamente: {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()