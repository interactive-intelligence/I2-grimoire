# I2 Grimoire

This is the repository for hosting the source for the I2 Grimoire!

[View the compiled PDF here!](https://grimoire.uw-i2.org)

## Notes

The web version is processed using kramdown before any math processing happens. ***Kramdown doesn't understand math*** and incorrectly interprets math as markdown. Therefore, one should use the following replacements in equations:

| Character | Replacement |
| --------- | ----------- |
| `\|`      | `\vert`     |
| `*`       | `\ast`      |
