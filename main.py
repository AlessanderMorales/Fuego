import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# FUNCIONES DEL MODELO
# ============================================================================

def tasa_propagacion(viento, humedad, pendiente, temperatura):
    """ Calcula la tasa de propagacion """
    R0 = 1.5 * (1.0 - 0.5 * humedad) * (1.0 + 0.05 * (temperatura - 20))
    v_magnitud = np.sqrt(viento[0]**2 + viento[1]**2)
    factor_viento = 1.0 + 2.0 * v_magnitud
    factor_pendiente = 1.0 + np.sin(np.radians(pendiente)) * 1.5
    return R0 * factor_viento * factor_pendiente

def termino_fuente(T, T_ignicion, T_quemado, rho_combustible):
    """ Generacion de calor por combustion """
    quemando = (T >= T_ignicion) & (rho_combustible > 0)
    Q = np.zeros_like(T)
    # Factor (1 - T/T_quemado) modela la pérdida de eficiencia al acercarse al máximo
    Q[quemando] = 10.0 * rho_combustible[quemando] * (1.0 - T[quemando] / T_quemado)
    return Q

def actualizar_combustible(rho_combustible, T, T_ignicion, dt):
    """ Consumo de combustible """
    quemando = T >= T_ignicion
    consumo = np.zeros_like(rho_combustible)
    consumo[quemando] = 0.2 * dt
    return np.maximum(rho_combustible - consumo, 0)

def propagacion_incendio(T, combustible, viento, pendiente_x, pendiente_y,
                        humedad, temperatura, dx, dy, dt, params):
    """ Propagacion del fuego """
    T_ignicion = params['T_ignicion']
    T_quemado = params['T_quemado']
    D_calor = params['D_calor']
    
    ny, nx = T.shape
    T_new = T.copy()
    
    # Difusion termica (Diferencias finitas)
    T_new[1:-1, :] += D_calor * dt / dx**2 * (T[2:, :] - 2*T[1:-1, :] + T[:-2, :])
    T_new[:, 1:-1] += D_calor * dt / dy**2 * (T[:, 2:] - 2*T[:, 1:-1] + T[:, :-2])
    
    # Propagacion a vecinos (Probabilística basada en viento)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if T[i, j] >= T_ignicion:
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    vi, vj = i + di, j + dj
                    if 0 <= vi < ny and 0 <= vj < nx:
                        prob = 0.1
                        # Efecto del viento en la probabilidad
                        if viento[0] > 0 and dj > 0: prob += 0.15 # Viento a favor en X
                        if viento[0] < 0 and dj < 0: prob += 0.15
                        if viento[1] > 0 and di > 0: prob += 0.15 # Viento a favor en Y
                        if viento[1] < 0 and di < 0: prob += 0.15
                        
                        if combustible[vi, vj] > 0 and np.random.random() < prob:
                            T_new[vi, vj] = max(T_new[vi, vj], T_ignicion + 50)
                            
    # Combustion y enfriamiento
    Q = termino_fuente(T_new, T_ignicion, T_quemado, combustible)
    T_new += Q * dt
    T_new -= 0.01 * (T_new - 20) * dt # Enfriamiento ambiental
    
    return T_new

# ============================================================================
# GENERACION DE ESCENARIO
# ============================================================================

def generar_escenario():
    """ Genera el escenario de simulacion """
    nx, ny = 60, 60
    Lx, Ly = 100.0, 100.0
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # Terreno con colina central
    Z = 10 * np.exp(-((X-50)**2 + (Y-50)**2) / 1000)
    pendiente_y, pendiente_x = np.gradient(Z, y[1]-y[0], x[1]-x[0])
    
    # Combustible en anillos (Ejercicio 3)
    combustible = np.zeros((ny, nx))
    centro_y, centro_x = ny // 2, nx // 2
    for i in range(ny):
        for j in range(nx):
            dist = np.sqrt(((i - centro_y) * Ly / ny)**2 + ((j - centro_x) * Lx / nx)**2)
            if dist < 15: combustible[i, j] = 10.0
            elif dist < 25: combustible[i, j] = 7.0
            elif dist < 35: combustible[i, j] = 4.0
            else: combustible[i, j] = 1.0
            
    return X, Y, pendiente_x, pendiente_y, combustible, Lx, Ly

def iniciar_fuego(ny, nx):
    """ Inicia el fuego en el centro """
    T = np.ones((ny, nx)) * 20.0
    centro_y, centro_x = ny // 2, nx // 2
    radio = 3
    for i in range(centro_y - radio, centro_y + radio):
        for j in range(centro_x - radio, centro_x + radio):
            if 0 <= i < ny and 0 <= j < nx:
                dist = np.sqrt((i - centro_y)**2 + (j - centro_x)**2)
                if dist < radio:
                    T[i, j] = 600.0
    return T

# ============================================================================
# VISUALIZACION
# ============================================================================

def crear_colormap_incendio():
    """ Colormap personalizado """
    colores = [
        (0.0, 'darkgreen'),
        (0.2, 'green'),
        (0.35, 'yellowgreen'),
        (0.5, 'yellow'),
        (0.65, 'orange'),
        (0.8, 'red'),
        (1.0, 'darkred')
    ]
    return LinearSegmentedColormap.from_list('incendio', colores)

def simular_incendio():
    print("=" * 70)
    print(" SIMULACION DE INCENDIO FORESTAL 2D")
    print("=" * 70)
    
    # Parametros base
    params = {
        'T_ignicion': 250.0,
        'T_quemado': 600.0,
        'D_calor': 0.2,
        'humedad': 0.15,
        'temperatura_amb': 25.0,
        'dt': 0.3,
        'viento_x': 3.0,
        'viento_y': 1.0
    }
    
    X, Y, pendiente_x, pendiente_y, combustible, Lx, Ly = generar_escenario()
    ny, nx = combustible.shape
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)
    
    print(f"\nPARAMETROS:")
    print(f" Dominio: {Lx:.0f} x {Ly:.0f} m")
    print(f" Viento: ({params['viento_x']}, {params['viento_y']}) m/s")
    print(f" Humedad: {params['humedad']*100}%")
    print(f" T ignicion: {params['T_ignicion']} C")
    
    T = iniciar_fuego(ny, nx)
    combustible_actual = combustible.copy()
    
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    cmap_fuego = crear_colormap_incendio()
    
    # Panel 1: Temperatura
    im1 = ax1.imshow(T, cmap=cmap_fuego, vmin=20, vmax=600, extent=[0, Lx, 0, Ly], origin='lower')
    ax1.set_title('Temperatura del Fuego', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='C')
    
    # Circulos de referencia
    centro_x, centro_y = Lx/2, Ly/2
    for radio in [15, 25, 35]:
        circle = plt.Circle((centro_x, centro_y), radio, fill=False, color='white', linestyle='--', alpha=0.5)
        ax1.add_patch(circle)
        
    # Panel 2: Combustible
    im2 = ax2.imshow(combustible, cmap='YlGn', vmin=0, vmax=10, extent=[0, Lx, 0, Ly], origin='lower')
    ax2.set_title('Combustible Restante', fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Cantidad')
    
    # Panel 3: Frente de llama
    im3 = ax3.imshow(T >= params['T_ignicion'], cmap='Reds', vmin=0, vmax=1, extent=[0, Lx, 0, Ly], origin='lower')
    ax3.set_title('Frente de Llama Activo', fontweight='bold')
    
    # Flecha de viento
    ax3.arrow(80, 80, params['viento_x']*5, params['viento_y']*5, head_width=3, head_length=2, fc='blue', ec='blue', alpha=0.8)
    ax3.text(80 + params['viento_x']*5 + 2, 80 + params['viento_y']*5 + 2, 'Viento', color='blue', fontsize=9)
    
    # Panel 4: Estadisticas
    ax4.set_title('Estadisticas en Tiempo Real', fontweight='bold')
    ax4.axis('off')
    stats_text = ax4.text(0.1, 0.5, '', transform=ax4.transAxes, fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, color='white', fontweight='bold', fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    def animate(frame):
        nonlocal T, combustible_actual
        for _ in range(3): # Acelerar simulacion
            T = propagacion_incendio(T, combustible_actual, (params['viento_x'], params['viento_y']), 
                                    pendiente_x, pendiente_y, params['humedad'], 
                                    params['temperatura_amb'], dx, dy, params['dt'], params)
            combustible_actual = actualizar_combustible(combustible_actual, T, params['T_ignicion'], params['dt'])
            
        t_actual = (frame + 1) * 3 * params['dt']
        im1.set_array(T)
        im2.set_array(combustible_actual)
        frente = (T >= params['T_ignicion']).astype(float)
        im3.set_array(frente)
        
        area_quemada = np.sum(frente) / frente.size * 100
        temp_max = T.max()
        
        time_text.set_text(f't = {t_actual:.1f} s')
        stats_text.set_text(f'Area quemada: {area_quemada:.1f} %\n'
                           f'T maxima: {temp_max:.0f} C\n'
                           f'Combustible medio: {combustible_actual.mean():.1f}')
        
        return [im1, im2, im3, time_text, stats_text]

    anim = animation.FuncAnimation(fig, animate, frames=150, interval=100, blit=True)
    print("Animacion lista! La ventana se abrira en unos segundos...")
    plt.show()
    return anim

if __name__ == "__main__":
    simular_incendio()