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
.PHONY: help check build test gpu smoke up down stop restart logs ps shell backup

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-9s %s\n", $$1, $$2}'

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
	  set -e; A="--runs-dir /tmp/smoke --name smoke --n-envs 2 --eval-every 1 --checkpoint-every 1"; \
	  python -m agents.ppo.run $$A --updates 2; \
	  python -m agents.ppo.run $$A --updates 3 --resume; \
	  python -m agents.ppo.run $$A --updates 3 --resume | grep -q "nothing to do"; \
	  ls /tmp/smoke/smoke/checkpoints; echo "smoke OK"'

up: build  ## start training + TensorBoard + Optuna dashboard in the background
	@test -f .env || { echo "no .env: cp .env.example .env and edit it"; exit 1; }
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
