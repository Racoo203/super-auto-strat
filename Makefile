# Server workflow for the PPO trainer. `make help` lists the targets.
# Docker targets need Docker + the NVIDIA Container Toolkit (deploy/README.md).

SHELL := /bin/bash
-include .env

export GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)$(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
export UID := $(shell id -u)
export GID := $(shell id -g)

COMPOSE := docker compose
RUN := $(COMPOSE) run --rm --no-deps

.DEFAULT_GOAL := help
.PHONY: help check build test gpu smoke up down stop restart logs ps shell backup \
        tune tune-logs tune-stop validate validate-logs validate-stop

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

# tune / validate run as one-off containers named sas-tune / sas-validate (the train service's
# image, GPU and shm settings). They share the GPU and CPU workers with `train`, so only one of
# train / tune / validate runs at a time. The exited container is kept (for `docker logs`) until
# the next `make tune` / `make validate` replaces it.
JOBS := sas-tune sas-validate
job_running = $$(for c in $(JOBS); do [ "$$(docker inspect -f '{{.State.Running}}' $$c 2>/dev/null)" = true ] && echo $$c; done)
define require_idle
	@if [ -n "$$($(COMPOSE) ps -q --status running train 2>/dev/null)" ]; then \
	  echo "train is running and would share the GPU/CPU: 'make stop' first"; exit 1; fi
	@running="$(job_running)"; if [ -n "$$running" ]; then \
	  echo "already running: $$running ('make tune-stop' / 'make validate-stop' first)"; exit 1; fi
endef

check:  ## verify the host: driver, docker, GPU inside containers
	bash deploy/check-host.sh

build:  ## build the image, tagged with the current commit
	mkdir -p runs
	$(COMPOSE) build train

test: build  ## run the test suite inside the image
	$(RUN) train python -m pytest -q -p no:cacheprovider

gpu: build  ## show which device training will use inside the container
	$(RUN) train python -c "import torch; from agents.ppo.config import get_device; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| device', get_device(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

smoke: build  ## end-to-end on the GPU: train 2 updates, resume to 3, re-resume is a no-op (nothing kept)
	$(RUN) train bash -c '\
	  set -e; A="--runs-dir /tmp/smoke --name smoke --n-envs 2 --eval-every 1 --eval-episodes 2 --checkpoint-every 1"; \
	  python -m agents.ppo.run $$A --updates 2; \
	  python -m agents.ppo.run $$A --updates 3 --resume; \
	  python -m agents.ppo.run $$A --updates 3 --resume | grep -q "nothing to do"; \
	  ls /tmp/smoke/smoke/checkpoints; echo "smoke OK"'

up: build  ## start training + TensorBoard + Optuna dashboard in the background
	@test -f .env || { echo "no .env: cp .env.example .env and edit it"; exit 1; }
	@running="$(job_running)"; if [ -n "$$running" ]; then \
	  echo "$$running is running and would share the GPU/CPU: 'make tune-stop' / 'make validate-stop' first"; exit 1; fi
	$(COMPOSE) up -d
	@echo "TensorBoard: http://$(or $(BIND_ADDR),127.0.0.1):$(or $(TB_PORT),6006)   Optuna: http://$(or $(BIND_ADDR),127.0.0.1):$(or $(OPTUNA_PORT),8080)"

stop:  ## stop training gracefully (saves latest.pt); dashboards keep running
	$(COMPOSE) stop train

restart: stop  ## stop and resume training (e.g. after `git pull` + rebuild)
	$(MAKE) up

down:  ## stop everything
	$(COMPOSE) down

logs:  ## follow training output
	$(COMPOSE) logs -f --tail 100 train

ps:  ## container status
	$(COMPOSE) ps

shell: build  ## interactive shell in the image (replay/watch tools, etc.)
	$(RUN) train bash

backup:  ## archive runs/ (checkpoints, metrics, optuna.db) to backups/
	mkdir -p backups
	tar czf backups/runs-$$(date +%Y%m%d-%H%M%S).tar.gz runs
	@ls -lh backups | tail -3

tune: build  ## hyperparameter search in the background (study ppo-tune; TUNE_ARGS in .env)
	$(require_idle)
	@docker rm sas-tune >/dev/null 2>&1 || true
	$(COMPOSE) up -d tensorboard optuna-dashboard
	$(COMPOSE) run -d --name sas-tune --no-deps train \
	  python -m agents.ppo.tune --n-envs $(or $(N_ENVS),8) $(TUNE_ARGS)
	@echo "started sas-tune: 'make tune-logs' to follow, 'make tune-stop' to stop"

tune-logs:  ## follow the hyperparameter search
	docker logs -f --tail 100 sas-tune

tune-stop:  ## stop the search gracefully (running trial -> FAIL; 'make tune' continues the study)
	docker stop -t 120 sas-tune

validate: build  ## rerun the top tuning trials with more seeds (VALIDATE_ARGS in .env)
	$(require_idle)
	@docker rm sas-validate >/dev/null 2>&1 || true
	$(COMPOSE) up -d tensorboard optuna-dashboard
	$(COMPOSE) run -d --name sas-validate --no-deps train \
	  python -m agents.ppo.validate --n-envs $(or $(N_ENVS),8) $(VALIDATE_ARGS)
	@echo "started sas-validate: 'make validate-logs' to follow, 'make validate-stop' to stop"

validate-logs:  ## follow the validation runs (summary table at the end)
	docker logs -f --tail 100 sas-validate

validate-stop:  ## stop validation gracefully ('make validate' continues where it stopped)
	docker stop -t 120 sas-validate
