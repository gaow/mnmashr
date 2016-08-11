.PHONY: clean install docs

install:
	#@R CMD check ./ --no-manual -o $(shell mktemp -d tmp.XXXX)
	@R CMD build ./ --no-manual
	@((R CMD INSTALL m2ashr_*.tar.gz -l $(shell echo "cat(.libPaths()[1])" | R --slave) && rm -rf tmp.* m2ashr_*.tar.gz) || ($(ECHO) "Installation failed"))

docs:
	@echo 'roxygen2::roxygenise()' | R --vanilla --silent

clean:
	rm -f src/m2ash.o src/m2ashr.so src/symbols.rds m2ashr_*.tar.gz
