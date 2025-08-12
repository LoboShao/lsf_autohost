############################################################################
#
# Load Sharing Applications
#
# Makefile for auto host assignment scheduler plugin
#
# Build: 
#     all:   build all objects
#     clear clean: remove all object files
# 
###########################################################################


#################### Testing purpose: by Yiming Shao#######################

# INCDIR = /opt/ibm/lsf/10.1/include
# LIBDIR = /opt/ibm/lsf/10.1/linux3.10-glibc2.17-x86_64/lib


###########################################################################


TOP= ../..
include $(TOP)/Make.misc

# LSF libraries needed

# Compiling flags 
INCLUDEPATH = -I../../../include/lsf
CFLAGS=${EXTERN_CFLAGS} ${INCLUDEPATH} ${SITE} ${CFLAG_SHLIB}

# Add CURL library for ML server communication
LIBS = -lcurl

#####################  Object files #########################
AUTOHOSTOBJS = auto_host.$(OEXT)



PLUGINAUTOHOST = schmod_auto_host.$(SOEXT)

# build
build all: $(PLUGINAUTOHOST)


# build objects
.c.$(OEXT):   $<
	${CC} ${CFLAGS} ${SHLIB_CFLAGS} -c -o $@ $<

.c.$(STATIC_OEXT):   $<
	${CC} -DSTATIC_LINK ${CFLAGS} -c -o $@ $<

.c.i:  $<
	${CC} ${CFLAGS} -C -E -c -o $@ $<

###################  Binary & Release #######################
${PLUGINAUTOHOST}: ${AUTOHOSTOBJS}
	$(SHLD) ${AUTOHOSTOBJS} ${LIBS} -o ${PLUGINAUTOHOST}

# clean up all object files and temporary files
clean clear:
	-rm -f *.$(OEXT) *.$(SOEXT) .\#* 

# Optional: install target
install: $(PLUGINAUTOHOST)
	@echo "Installing plugin to LSF plugin directory..."
	# cp $(PLUGINAUTOHOST) $(LSF_LIBDIR)/plugins/
	@echo "Plugin built successfully: $(PLUGINAUTOHOST)"

# Optional: debug build
debug: CFLAGS += -g -DDEBUG
debug: $(PLUGINAUTOHOST)

.PHONY: build all clean clear install debug