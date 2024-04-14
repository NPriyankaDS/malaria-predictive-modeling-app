DHSID = The 14 character DHS identification code - DHSCC & DHSYEAR & DHSCLUST (with 8 digits) from survey documentation.

DHSCC = The 2 letter DHS country code (http://www.dhsprogram.com/data/File-Types-and-Names.cfm).

DHSYEAR = The 4 digit year of data collection from the survey documentation.

DHSCLUST = The integer cluster identification number. This variable will match v001 in the DHS recode file. 

CCFIPS = Federal Information Processing Standards (FIPS) 2 letter country code (http://www.itl.nist.gov/fipspubs/fip10-4.htm).

ADM1FIPS = Federal Information Processing Standards (FIPS) 2  letter country code plus 2 letter/digit first sub-national administrative division code (http://www.itl.nist.gov/fipspubs/fip10-4.htm).
*NOTE: If this information is not available, this field will be "NULL".

ADM1FIPSNA = Federal Information Processing Standards (FIPS) first sub-national administrative division name (http://www.itl.nist.gov/fipspubs/fip10-4.htm).
*NOTE: If this information is not available, this field will be "NULL".

ADM1SALBCO = Second Administrative Level Boundaries (SALB) first sub-national administrative division code (http://www.unsalb.org).
*NOTE: The website requires free registration for downloads.
*NOTE: If this information is not available, this field will be "NULL".

ADM1SALBNA = Second Administrative Level Boundaries (SALB) first sub-national administrative division name (http://www.unsalb.org). 
*NOTE: The website requires free registration for downloads.
*NOTE: If this information is not available, this field will be "NULL".

ADM1DHS = First sub-national administrative division code when the DHS sample is representative at the admin 1 level. This variable will usually match v024 in the DHS recode file.
*NOTE: If survey is not representative at the admin 1 level, this field will be "9999".

ADM1NAME = First sub-national administrative division name when the DHS sample is representative at the admin 1 level. This variable will usually match v024 in the DHS recode file.
*NOTE: If survey is not representative at the admin 1 level, this field will be "NULL".

DHSREGCO = The integer region code associated with the DHS region created for sampling. This variable will match either v024 or the country specific region variable in the DHS recode file.
*NOTE:  In older templates, REPAR1DHS was used. This field has been renamed DHSREGCO. The REPAR1DHS field is no longer used. 

DHSREGNA = The name associated with the DHS region created for sampling.  This variable will match either v024 or the country specific region variable in the DHS recode file.
*NOTE:  In older templates, REPAR1NAME was used. This field has been renamed DHSREGNA. The REPAR1NAME field is no longer used. 

SOURCE = The source of data used to determine the latitude and longitude coordinates:
“GPS” for data collected by the survey team with a global positioning system receiver;
"CEN" for preexisting data provided by the census agency/ministry;
“GAZ” for data extracted from a gazetteer of village/place names;
“MAP” for data extracted from a paper map;
"MIS" for clusters in which data could not be fully verified. Clusters marked as "MIS" will have coordinates 0, 0.

URBAN_RURA = The cluster's Urban (U) and Rural (R) DHS sample classification.

LATNUM = The cluster's latitude coordinate in decimal degrees.
*NOTE:  Clusters marked as "MIS" will have coordinates of 0, 0.

LONGNUM = The cluster's longitude coordinate in decimal degrees.
*NOTE:  Clusters marked as "MIS" will have coordinates of 0, 0.

ALT_GPS = The cluster's elevation/altitude (in meters) recorded from the GPS receiver.
*NOTE: If this information is not available, this field will be "9999".

ALT_DEM = The cluster's elevation/altitude (in meters) from the SRTM (Shuttle Radar Topography Mission) DEM (Digital Elevation Model) for the specified coordinate location.
*NOTE: Elevations are regularly spaced at 30-arc seconds or approximately 1 kilometer (http://dds.cr.usgs.gov/srtm/version1/SRTM30).
*NOTE: If coordinates are missing, this field will be "9999".

DATUM = The coordinate reference system and geographic datum. It is always "WGS84" for the World Geodetic System (WGS) 1984.