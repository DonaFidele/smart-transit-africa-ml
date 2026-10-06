# Protocole de comptage terrain – SmartTransit Africa (Cotonou)

## 1. Objectif
Produire un petit jeu de données **réel** pour deux validations :
1. Le profil de chaque site (niveau de trafic, pointes, soirée) correspond-il à l'archétype supposé ?
2. Les alertes de saturation du modèle sont-elles cohérentes avec ce qu'on observe ?

Il s'agit d'une validation **exploratoire** : quelques dizaines d'observations ne prouvent rien de façon définitive.

## 2. Sites
Les 5 sites de l'application : Godomey, Dantokpa, Vèdoko, Cadjehoun, Zone portuaire / Akpakpa.
Si tu ne peux pas tous les couvrir, prends-en **au moins 3** (nécessaire pour comparer les sites entre eux).

## 3. Calendrier minimal
- **2 jours** par site : un jour de marché (mercredi ou samedi) et un jour ordinaire.
- **3 créneaux d'une heure par jour** : pointe du matin (7 h–9 h), milieu de journée (12 h–14 h), pointe du soir (16 h–19 h).
- Soit **6 fenêtres par site, 30 au total pour 5 sites**. Plus tu en fais, plus la validation est solide (idéal : 60 et plus).
- Un créneau de soirée (20 h–22 h) est un plus, s'il est sûr.
- **Pas de comptage de nuit en solitaire.** Ne prends aucun risque : la sécurité passe avant les données.

## 4. Méthode de comptage
1. Choisis un **point fixe** par site (un repère visible) et **un sens de circulation** ; garde-les identiques pour tous les comptages.
2. Compte chaque véhicule qui franchit le repère pendant la fenêtre. Utilise un compteur manuel ou une application, un compteur par catégorie.
3. Catégories : `motos` (zémidjans et motos), `voitures`, `bus_minibus` (minibus, taxis collectifs, bus), `camions`.
4. Une ligne du fichier = une fenêtre (par exemple 07:00 → 08:00) avec le total par catégorie.
5. Si tu ne couvres que 45 minutes, renseigne les heures réelles : le fichier normalise en véhicules par heure.

## 5. « Saturation observée » (0 ou 1)
**Fixe la définition avant de commencer, et garde-la pour tout le projet.** Proposition :

> `1` si, au moins une fois pendant la fenêtre, la file d'attente dépasse un repère fixé à l'avance (par exemple 100 m en amont du point de comptage) **ou** si des véhicules mettent plus de 2 minutes à franchir le carrefour. Sinon `0`.

Note le repère choisi pour chaque site dans la colonne `notes` de la première ligne.

## 6. Pluie
Colonne `pluie_mm` (estimation de l'intensité pendant la fenêtre) :
`0` = sec ; `1` = bruine ; `3` = pluie modérée ; `8` = forte pluie. Tu pourras comparer ensuite avec les données météo d'Open-Meteo.

## 7. Format du fichier
Colonnes obligatoires, séparateur virgule, dates au format `AAAA-MM-JJ`, heures `HH:MM` :

```
date,site,heure_debut,heure_fin,motos,voitures,bus_minibus,camions,pluie_mm,saturation_observee,observateur,notes
```

Exemple (valeurs **fictives**, à ne pas réutiliser) :

```
2026-10-14,Dantokpa,07:00,08:00,1250,430,95,30,0,1,A,repère 100 m
```

La colonne `site` accepte le nom complet ou un mot-clé : `Godomey`, `Dantokpa`, `Vedoko`, `Cadjehoun`, `Portuaire`.
Le modèle de fichier vide se télécharge aussi depuis l'onglet « Field Validation » de l'application.

## 8. Éthique et pratique
- Compte sans filmer ni photographier les personnes ; ne note aucune donnée personnelle.
- Préviens, si nécessaire, la mairie ou la police locale pour éviter tout malentendu.
- Place-toi à un endroit sûr, hors de la chaussée.

## 9. Comment présenter les résultats
> « Un comptage terrain exploratoire (N fenêtres, K sites) a servi à tester la cohérence du transfert. Les résultats sont indicatifs et appellent une collecte plus large. »

Si les résultats contredisent l'hypothèse (écart « divergent », modèle peu cohérent), c'est un **résultat à discuter**, pas un échec : il montre à quoi sert une validation locale.