images_names=kodim01.png kodim02.png

all: split
images:$(addprefix testImage/,$(images_names))
.PHONY: clean test

BINS=split calibrate tests/createBlankImage
COMMONOBJ:=lib/fillwithblack.o lib/manppm.o
OBJ:=SplitByCalibration.o CalibrateByColor.o $(COMMONOBJ)
EXTRACFLAGS=
CFLAGS=-Wall $(EXTRACFLAGS)

$(BINS):
	$(CC) $(CFLAGS) -lm -o $@ $^

split: SplitByCalibration.o $(COMMONOBJ)
calibrate: CalibrateByColor.o $(COMMONOBJ)
tests/createBlankImage: tests/createBlankImage.o lib/fillwithblack.o lib/manppm.o


testImage/kodim%.png:
	wget --quiet http://r0k.us/graphics/kodak/kodak/$(notdir $@) -O $@

clean:
	rm -f $(OBJ) split tests/createBlankImage



test: tests/cuttest tests/createBlankImage split
	cd $(dir $<) && bash $(notdir $<)
