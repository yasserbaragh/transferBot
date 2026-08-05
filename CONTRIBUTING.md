# Contributing

Thanks for wanting to help out. A few ground rules before you open a pull request:

- **Document your changes.** Explain what you changed and why in the pull request description. If it's not obvious from the code, it's not obvious to whoever reviews it either.
- **Don't change the database.** This project uses SQLite on purpose. Don't swap it for Postgres, MySQL, or anything else.
- **Don't remove existing sources.** The feeds listed in `config/sources.yaml` stay. Adding new ones is welcome, but don't delete existing ones.
- **Include a couple of test cases for any keyword/regex changes.** If you touch the relevance filters, show a string that should match and one that shouldn't, so a reviewer doesn't have to trace the regex by hand.
- **Flag any change to the extraction prompt or schema clearly.** These affect every rumor going forward, so call them out in the PR description instead of folding them quietly into an unrelated fix.

That's it for now.
