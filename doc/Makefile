# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first three.
SPHINXOPTS      ?=
SPHINXBUILD     ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR       = _build/src
BUILDDIR        = _build/

SPHINXOPTS := ${SPHINXOPTS} --conf-dir .

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

mkdir:
	mkdir -p ${SOURCEDIR}

clean:
	rm ${BUILDDIR} -r

# this live will not update on file change due to zephyr.external_content sphinx extension
live: mkdir
	@$(SPHINXAUTOBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: mkdir Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
