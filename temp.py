import wandb
api = wandb.Api()
run = api.run("nannuriabhi2000-hochschule-schmalkalden/dexmg-bc/kpid6s4r")
for art in run.logged_artifacts():
    print(art.name, art.state)