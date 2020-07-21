REPO_DIR=$(PWD)
INSTALL_DIR=$(HOME)/.phy/
PLUGINS_INSTALL_DIR=$(INSTALL_DIR)/plugins/
PLUGINS:=$(wildcard plugins/*.py)

MKDIR=mkdir -p $(PLUGINS_INSTALL_DIR)

simlink: clean phy_config.py $(PLUGINS)
	$(MKDIR)
	ln -s $(REPO_DIR)/phy_config.py $(INSTALL_DIR)
	$(foreach file,$(PLUGINS), ln -s $(REPO_DIR)/$(file) $(PLUGINS_INSTALL_DIR);)

copy: phy_config.py $(PLUGINS)
	$(MKDIR)
	cp phy_config.py $(INSTALL_DIR)
	cp $(foreach file,$(PLUGINS), $(REPO_DIR)/$(file)) $(PLUGINS_INSTALL_DIR)

clean: 
	rm -rf $(INSTALL_DIR)/phy_config.py
	rm -rf $(foreach file,$(PLUGINS), $(INSTALL_DIR)/$(file))
