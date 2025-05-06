# Simulación de un robot en el entorno Aloha

# Este script muestra cómo trabajar con el framework LeRobot para simular
# robots controlados por IA y evaluar su rendimiento.

# Instalar dependencias necesarias

# git clone https://github.com/huggingface/lerobot.git
# cd lerobot
# conda create -y -n lerobot python=3.10
# conda activate lerobot
# pip install -e ".[aloha]"
# pip install matplotlib

# Importar bibliotecas necesarias
import cv2
import gym_aloha  
import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt
import imageio
from pathlib import Path

# Importaciones específicas de LeRobot
from lerobot.common.policies.act.modeling_act import ACTPolicy


# Configurar directorio para guardar resultados
output_directory = Path("outputs/aloha_simulation")
output_directory.mkdir(parents=True, exist_ok=True)

# Seleccionar el dispositivo para inferencia (CUDA para GPU, MPS o CPU como fallback)
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Utilizando dispositivo: {device}") 
# Cargar un modelo preentrenado desde Hugging Face
print("Cargando modelo preentrenado...")
pretrained_policy_path = "lerobot/act_aloha_sim_transfer_cube_human"
policy = ACTPolicy.from_pretrained(pretrained_policy_path)
policy.to(device)

# Inicializar el entorno de simulación
print("Inicializando entorno de simulación Aloha...")
env = gym.make(
    "gym_aloha/AlohaTransferCube-v0",
    obs_type="pixels_agent_pos",  # Observación: Imagen de la escena + posición del agente
    max_episode_steps=400,        # Máximo número de pasos por episodio
)

# Verificar compatibilidad entre modelo y entorno
print("\nVerificando compatibilidad entre modelo y entorno:")
print("Entradas esperadas por el modelo:", policy.config.input_features)
print("Espacio de observación del entorno:", env.observation_space)

print("\nSalidas del modelo:", policy.config.output_features)
print("Espacio de acción del entorno:", env.action_space)

# Función para ejecutar un episodio en el entorno
def run_episode(env, policy, device, render=True, save_video=False):
    """
    Ejecuta un episodio completo utilizando la política preentrenada
    
    Args:
        env: Entorno de simulación
        policy: Política preentrenada
        device: Dispositivo de PyTorch (cuda/cpu)
        render: Si se debe mostrar cada paso
        save_video: Si se debe guardar un video del episodio
    
    Returns:
        dict: Resultados del episodio con métricas
    """
    # Reiniciar el entorno y la política
    policy.reset()
    observation, info = env.reset(seed=43) # esta semilla no se queda trabada
    
    # Preparar colección de frames si guardamos video
    frames = []
    if render or save_video:
        frames.append(env.render())
    
    # Colectar métricas
    rewards = []
    done = False
    step = 0
    
    # Bucle principal del episodio
    while not done:
        # Preparar observación para la política
        state = torch.from_numpy(observation["agent_pos"]).to(torch.float32)
        image = torch.from_numpy(observation["pixels"]["top"]).to(torch.float32) / 255.0
        image = image.permute(2, 0, 1)  # Cambiar a formato canal-primero
        
        # Mover tensores a GPU/CPU
        state = state.to(device)
        image = image.to(device)
        
        # Añadir dimensión de lote (batch)
        state = state.unsqueeze(0)
        image = image.unsqueeze(0)
        
        # Crear diccionario de entrada para la política
        observation_dict = {
            "observation.state": state,
            "observation.images.top": image,
        }
        
        # Predecir acción usando la política
        with torch.inference_mode():
            action = policy.select_action(observation_dict)
        
        # Convertir a formato numpy para el entorno
        numpy_action = action.squeeze(0).cpu().numpy()
        
        # Aplicar acción en el entorno
        observation, reward, terminated, truncated, info = env.step(numpy_action)
        
        # Guardar frame si estamos creando video
        if save_video:
            frames.append(env.render())
        
        # Actualizar métricas
        rewards.append(reward)
        
        # Verificar si el episodio terminó
        done = terminated or truncated
        step += 1
    
    # Guardar video si se solicitó
    if save_video:
        video_path = output_directory / "episodio_aloha.mp4"
        fps = env.metadata.get("render_fps", 30)
        
        # Encuentra las dimensiones divisibles por 16 más cercanas
        h, w = frames[0].shape[:2]
        new_h = ((h + 15) // 16) * 16  # Redondea hacia arriba a múltiplo de 16
        new_w = ((w + 15) // 16) * 16

        # Redimensiona todos los frames
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (new_w, new_h))
            resized_frames.append(resized_frame)

        # Guarda el video con los frames redimensionados
        imageio.mimsave(str(video_path), np.stack(resized_frames), fps=fps)
        print(f"Video guardado en: {video_path}")
    
    # Preparar y devolver resultados
    results = {
        "pasos_totales": step,
        "recompensa_total": sum(rewards),
        "exito": terminated,  # En PushT, terminated significa éxito
        "recompensas": rewards,
    }
    
    return results

# Ejecutar un episodio y guardar video
print("\nEjecutando episodio de simulación...")
results = run_episode(env, policy, device, render=False, save_video=True)

# Mostrar resultados
print("\nResultados del episodio:")
print(f"Pasos totales: {results['pasos_totales']}")
print(f"Recompensa total: {results['recompensa_total']:.3f}")
print(f"Éxito en la tarea: {'Sí' if results['exito'] else 'No'}")

# Visualizar la evolución de la recompensa durante el episodio
plt.figure(figsize=(10, 5))
plt.plot(results['recompensas'])
plt.title('Evolución de la recompensa durante el episodio')
plt.xlabel('Paso')
plt.ylabel('Recompensa')
plt.grid(True)
plt.show()

# -----------------------------------------------------------------
# Experimento: Modificar la posición inicial del objetivo
# -----------------------------------------------------------------

print("\nExperimento: Evaluando la robustez del modelo ante cambios en el entorno")

# Función para ejecutar múltiples episodios con diferentes semillas
def run_multiple_episodes(env, policy, device, num_episodes=5):
    """Ejecuta múltiples episodios con diferentes semillas"""
    success_rate = 0
    total_rewards = []
    
    for ep in range(num_episodes):
        # Usar diferentes semillas para cada episodio
        seed = 42 + ep
        observation, info = env.reset(seed=seed)
        
        policy.reset()
        rewards = []
        done = False
        step = 0
          
        while not done:
            # Preparar observación para la política (igual que antes)
            state = torch.from_numpy(observation["agent_pos"]).to(torch.float32)
            image = torch.from_numpy(observation["pixels"]["top"]).to(torch.float32) / 255.0
            image = image.permute(2, 0, 1)
            
            state = state.to(device).unsqueeze(0)
            image = image.to(device).unsqueeze(0)
            
            observation_dict = {
                "observation.state": state,
                "observation.images.top": image,
            }
            
            with torch.inference_mode():
                action = policy.select_action(observation_dict)
            
            numpy_action = action.squeeze(0).cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(numpy_action)
            
            rewards.append(reward)
            done = terminated or truncated
            step += 1
        
        # Actualizar métricas
        success_rate += int(terminated)  # terminated = éxito en PushT
        total_rewards.append(sum(rewards))
        
        print(f"Episodio {ep+1}: {'Éxito' if terminated else 'Fracaso'} | Recompensa: {sum(rewards):.3f} | Pasos: {step}")
    
    # Calcular métricas finales
    success_rate = success_rate / num_episodes * 100
    avg_reward = sum(total_rewards) / num_episodes
    
    return {
        "tasa_exito": success_rate,
        "recompensa_promedio": avg_reward,
        "recompensas_episodios": total_rewards
    }

# Ejecutar múltiples episodios para evaluar la robustez
results_experiment = run_multiple_episodes(env, policy, device, num_episodes=5)

# Mostrar resultados del experimento
print("\nResultados del experimento:")
print(f"Tasa de éxito: {results_experiment['tasa_exito']}%")
print(f"Recompensa promedio: {results_experiment['recompensa_promedio']:.3f}")

# Visualizar recompensas por episodio
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(results_experiment['recompensas_episodios'])+1), results_experiment['recompensas_episodios'])
plt.title('Recompensa total por episodio')
plt.xlabel('Episodio')
plt.ylabel('Recompensa total')
plt.xticks(range(1, len(results_experiment['recompensas_episodios'])+1))
plt.grid(True, axis='y')
plt.show()

# -----------------------------------------------------------------
# Resumen y conclusiones
# -----------------------------------------------------------------

print("\n---- Conclusiones ----")
print("1. Hemos evaluado un modelo de ACT Policy en el entorno Aloha")
print(f"2. El modelo consigue una tasa de éxito del {results_experiment['tasa_exito']}% con diferentes configuraciones iniciales")
print("3. La recompensa acumulada varía entre episodios, lo que indica sensibilidad a las condiciones iniciales")

# Cerrar el entorno al finalizar
env.close()
