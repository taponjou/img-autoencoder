# 🎨 Image Autoencoder (`img-autoencoder`)

Ce projet met en œuvre un modèle d'**Auto-encodeur Convolutionnel (Convolutional Autoencoder - CAE)** conçu pour la compression, la reconstruction et la réduction de dimensionnalité d'images. 

Il permet d'apprendre des représentations compressées (espace latent) à partir d'un ensemble d'images d'entrée, puis de reconstruire ces images avec une perte d'information minimale.

---

## 📌 Table des matières

* [Aperçu de l'Architecture](#-aperçu-de-larchitecture)
* [Fonctionnalités](#-fonctionnalités)
* [Structure du Projet](#-structure-du-projet)
* [Installation](#-installation)
* [Utilisation](#-utilisation)
  * [1. Entraînement](#1-entraînement)
  * [2. Évaluation et Reconstruction](#2-évaluation-et-reconstruction)
  * [3. Visualisation du Space Latent](#3-visualisation-du-space-latent)
* [Résultats](#-résultats)
* [Licence](#-licence)

---

## 🏗 Aperçu de l'Architecture

Le modèle repose sur une structure **Encoder-Decoder** classique :

1. **Encodeur (Encoder) :** 
   * Traite l'image d'entrée $X \in \mathbb{R}^{H \times W \times C}$ à travers plusieurs couches de convolutions et de max-pooling.
   * Compresse l'information vers un **vecteur latent (bottleneck)** $z \in \mathbb{R}^d$.

2. **Décodeur (Decoder) :**
   * Reçoit le vecteur latent $z$.
   * Utilise des couches de convolutions transposées (ou upsampling + convolution) pour reconstruire l'image d'origine $\hat{X} \in \mathbb{R}^{H \times W \times C}$.

$$\text{Loss} = \mathcal{L}_{MSE}(X, \hat{X}) = \frac{1}{N} \sum_{i=1}^{N} (X_i - \hat{X}_i)^2$$

---

## ✨ Fonctionnalités

* 🔄 **Reconstruction d'images :** Entraînement sur images personnalisées ou jeux de données standards (MNIST, CIFAR-10, CelebA, etc.).
* 🧹 **Denoising Autoencoder (Optionnel) :** Capacité à supprimer le bruit ajouté aux images d'entrée.
* 📊 **Visualisation des vecteurs latents :** Projection t-SNE / PCA pour analyser la distribution des représentations.
* 💾 **Sauvegarde & Chargement :** Export automatique des checkpoints et des meilleures métriques de validation.

---

## 📁 Structure du Projet

```text
img-autoencoder/
├── data/                  # Dossier pour les jeux de données
├── notebook/              # Notebooks Jupyter pour l'exploration et les tests
│   └── main.ipynb
├── models/                # Définitions des architectures (PyTorch / TensorFlow)
│   ├── encoder.py
│   ├── decoder.py
│   └── autoencoder.py
├── checkpoints/           # Modèles entraînés (.pth / .h5)
├── utils/                 # Fonctions auxiliaires (prétraitement, plots)
├── train.py               # Script principal d'entraînement
├── evaluate.py            # Script d'évaluation
├── requirements.txt       # Dépendances Python
└── README.md

1. Entraînement
Pour lancer l'entraînement du modèle avec les paramètres par défaut :
python train.py --epochs 50 --batch_size 64 --lr 0.001

2. Évaluation et Reconstruction
Pour tester la reconstruction sur un lot d'images de test :
python evaluate.py --model_path checkpoints/best_model.pth
