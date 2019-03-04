# Putting ML to Production
This repo contains code that we hope is useful to illustrate how one could productionise a real-time algorithm.

The code in this repo is meant to be as generic as possible, and is designed to be useful for the following scenario

## Scenario

A company collects data using a series of services that generate events as the users/customers interact with the the company's website or app. As these interactions happen, an algorithm needs to run in real time and some immediate action needs to be taken based on the algorithm's outputs (or predictions). On top of that, after N interactions (or observations) the algorithm needs to be retrained without stopping the prediction service, since users will keep interacting.