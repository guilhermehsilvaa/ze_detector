import os
import shutil
import random

# Caminho para o diretório do dataset
dataset_dir = 'datasets'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

# Caminhos para os diretórios de saída
train_images_dir = os.path.join(dataset_dir, 'train', 'images')
train_labels_dir = os.path.join(dataset_dir, 'train', 'labels')
val_images_dir = os.path.join(dataset_dir, 'val', 'images')
val_labels_dir = os.path.join(dataset_dir, 'val', 'labels')

# Criar diretórios de saída
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Listar todos os arquivos de imagens
images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

# Embaralhar aleatoriamente as imagens
random.shuffle(images)

# Definir índices de divisão
train_size = int(0.8 * len(images))
train_images = images[:train_size]
val_images = images[train_size:]

# Função para mover os arquivos correspondentes de imagem e label
def move_files(images, source_images_dir, source_labels_dir, target_images_dir, target_labels_dir):
    for image in images:
        label = image.replace('.jpg', '.txt')  # Assumindo que os rótulos têm a mesma nomenclatura das imagens

        # Caminhos completos
        src_image_path = os.path.join(source_images_dir, image)
        src_label_path = os.path.join(source_labels_dir, label)
        dest_image_path = os.path.join(target_images_dir, image)
        dest_label_path = os.path.join(target_labels_dir, label)

        # Mover os arquivos
        shutil.move(src_image_path, dest_image_path)
        shutil.move(src_label_path, dest_label_path)

# Mover arquivos para o diretório de treino
move_files(train_images, images_dir, labels_dir, train_images_dir, train_labels_dir)

# Mover arquivos para o diretório de validação
move_files(val_images, images_dir, labels_dir, val_images_dir, val_labels_dir)

print(f'Divisão completa: {len(train_images)} imagens para treino e {len(val_images)} imagens para validação.')