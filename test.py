from chessmc.trainer import AdvancedTrainerConfig, AdvancedTrainer
cfg = AdvancedTrainerConfig()
trainer = AdvancedTrainer(cfg)
print('init OK; lr warmup epochs =', cfg.warmup_epochs)