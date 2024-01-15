create table `Profile`
	(userName			varchar(20),
     `password`			varchar(20),
     firstName			varchar(30),
     lastName			varchar(30),
     email				varchar(40),
     aboutMe            varchar(1000),
     primary key (userName)
	);
create table Location
	(locationID			int,
     locationName		varChar(40),
     zipCode			int,
     park				boolean,
     primary key (locationID)
	);
create table Activity
	(activityName		varchar(40),
     activityID   		int,
     gpsCoord			varchar(40),
     aType				varchar(40),
     difficulty			int,
     dateAdded			date,
     locationID			int,
     userName			varchar(20),
     primary key (activityID),
     foreign key (locationID) references Location(locationID)
		on delete set null,
	 foreign key (userName) references `Profile`(userName)
		on delete set null
	);
