Here are brief descriptions of the data.

Movies Metadata.csv FILE DESCRIPTION
================================================================================

Information about the items (movies); this is a comma-delimited list of
              movie_id_Alias , movie Genres , Movie_ReleaseYear

Note: Genres are pipe-separated and are selected from the following genres:

	* Action
	* Adventure
	* Animation
	* Children's
	* Comedy
	* Crime
	* Documentary
	* Drama
	* Fantasy
	* Film-Noir
	* Horror
	* Musical
	* Mystery
	* Romance
	* Sci-Fi
	* Thriller
	* War
	* Western


Users Demographics.csv FILE DESCRIPTION
================================================================================
Information about the users;  this is a comma-delimited list of
              User_ID_Alias, User_Age, User_Gender, User_Occupation, User_ZipCode

- Gender is denoted by a "M" for male and "F" for female
- Age is chosen from the following ranges:

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- Occupation is chosen from the following choices:

	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"


Train.csv FILE DESCRIPTION
================================================================================
859395 ratings by 6040 users on 3224 items
this is a comma-delimited list of
              User_ID_Alias, Movie_ID_Alias, Rating


Validation.csv FILE DESCRIPTION
================================================================================
6040 ratings by 6040 (1 movie per user)
this is a comma-delimited list of
              User_ID_Alias, Movie_ID_Alias, Rating


Test.csv FILE DESCRIPTION
================================================================================
6040 ratings of 6040 users to forecast (1 movie per user)
this is a comma-delimited list of
              User_ID_Alias, Movie_ID_Alias






















