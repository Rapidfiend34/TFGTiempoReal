from visdom import Visdom
import os

# Crear directorio para guardar (opcional)
save_dir = './visdom_saves'
os.makedirs(save_dir, exist_ok=True)

# Conectar a la sesión existente
vis = Visdom(port=28333)

# Guardar el estado actual en una ubicación específica
vis.save(['main'], save_dir + '/visdom_state.json')  # Especifica la ruta completa
print(f"Estado guardado en: {os.path.abspath(save_dir)}/visdom_state.json")