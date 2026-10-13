# AIORI-3-HACKATHON-PROJECTS

This repository will contain the code and resources for the AIORI-3 hackathon that will be developed during the AIORI-3





**## A3-PS003-TC008: Federated Traffic-Flow Prediction**



Problem statement PS-003: Federated Traffic-Flow Prediction with Communication-Cost Evaluation (AIORI Hackathon 3, Smart Cities).



Dataset: Kaggle "Traffic Prediction Dataset" (fedesoriano), 4 junctions, hourly vehicle counts.

Download traffic.csv from Kaggle and place it in the repository's top folder.



Run order (run from the top folder):

1\. python src/prepare\_data.py

2\. python src/train\_central.py 0   (also seeds 1, 2)

3\. python src/train\_fed.py noniid 0   (also iid, seeds 1, 2)

4\. python src/analyze.py



Graphs and summary table: see the results folder.

Pseudocode: see PSEUDOCODE.md

Intro video: (YouTube link will be added here)

