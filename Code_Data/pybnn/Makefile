
SRCDIR = src
OBJDIR = obj
BINDIR = bin

PYTHON ?= ../../.venv/bin/python
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
LDSHARED := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LDSHARED') or 'c++ -shared')")
PYBIND11_INCLUDES ?= $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)

TARGET_LIB = pybnn$(EXT_SUFFIX)

OPT = -O3 -march=native
CXX = c++

CXXFLAGS = -std=c++11 $(OPT) -Wall -fPIC -I. -I$(PYTHON_INCLUDE) $(PYBIND11_INCLUDES)

LDFLAGS = $(wordlist 2,$(words $(LDSHARED)),$(LDSHARED))
LDLIBS =

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

all: $(BINDIR)/$(TARGET_LIB)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/$(TARGET_LIB): $(OBJECTS) | $(BINDIR)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS)
	@echo "Built $@ successfully."

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< successfully."

clean:
	rm -f $(BINDIR)/$(TARGET_LIB)
	rm -f $(OBJDIR)/*.o
