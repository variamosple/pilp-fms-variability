#!/usr/bin/env bash
set -e

echo "[1/3] Adding official SWI-Prolog stable PPA and updating packages..."
sudo add-apt-repository -y ppa:swi-prolog/stable
sudo apt-get update -qq
sudo apt-get install -y -qq swi-prolog swi-prolog-nox swi-prolog-devel

echo "[2/3] Installing and compiling cplint and bddem packs..."
swipl -g "pack_install(cplint, [interactive(false), upgrade(true)]), halt."
swipl -g "pack_rebuild(bddem), halt."

echo "[3/3] Verifying SLIPCOVER, PITA, and BDDEM initialization..."
swipl -q -g "use_module(library(bddem)), use_module(library(slipcover)), use_module(library(pita)), writeln('CPLINT_ENVIRONMENT_READY'), halt."