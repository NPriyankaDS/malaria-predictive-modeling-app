Dataset: LBGE81FL
Displacement Restriction: admin2 (districts)


In order to ensure that respondent confidentiality is maintained, we randomly displace the GPS 
latitude/longitude positions for all DHS, MIS, and AIS surveys. 
The displacement is randomly carried out so that:
-Urban clusters are displaced up to 2 kilometers. 
-Rural clusters are displaced up to 5 kilometers, with 1% of the rural clusters displaced up to 10 kilometers.

Restricted to admin2:
The displacement is restricted so that the points stay within the country, within the DHS survey region, and within
the admin2 area. Therefore, the displaced cluster's coordinates are located within the same admin0, admin1,
and admin2 areas as the undisplaced cluster.


This random error can substantively affect analysis results, where analysis questions look at small geographic 
areas.  Specifically, measuring direct distance from a GPS location to some other site (Facility, school, etc) 
is NOT correct since this does not account for the displacement of the GPS locations.

For more information about how to account for displacement when using our GPS datasets, users may consult the two most recent
publications in our Spatial Analysis Reports series:

- Geographic Displacement Procedure and Georeferenced Data Release Policy for the Demographic and Health Surveys (English)
http://dhsprogram.com/publications/publication-SAR7-Spatial-Analysis-Reports.cfm

- Guidelines on the Use of DHS GPS Data (English)
http://dhsprogram.com/publications/publication-SAR8-Spatial-Analysis-Reports.cfm


A detailed description of the displacement procedure can be found below:

“Displacement” or geographically off-setting GPS Data in The Demographic and Health Surveys (DHS) Program

DHS surveys contain confidential information that could be used inappropriately to identify an individual through 
unique information and the location or place of residence.  To avoid this possibility, The DHS Program has developed 
an approach to degrade the accuracy of the GPS coordinates so that the true place of residence cannot be derived. 
This procedure will significantly reduce the likelihood of identifying an individual, yet still retain the locational 
detail for spatial analysis.

In all DHS surveys, the center of the populated place in the cluster is recorded with Global Positioning System receivers. 
These data have been collected and referenced in geographic coordinates (degrees in latitude and longitude).  
The circular error resulting from within the GPS system is less than 100 meters from the true location for each value.  
In practice, and especially during ideal GPS data collection situations (flat horizon, no obstructions from vegetation 
canopy or buildings), the coordinates are very robust and will typically have less than 10m of error. 

Applying an appropriate amount of error is desired, but somewhat subjective. Rural areas are usually less densely 
populated than urban areas, and the two regions are often considered separately during data analysis. Separate 
degradation error values for each are recommended.  They have been selected to be up to 5km for rural points 
(with 1% of rural cluster being randomly displaced up to 10km) and up to 2km for urban (all urban points have the 
same 2km randomization applied). Previous ICF International analyses have shown that applying random error of 5km 
maximum in rural areas and 2km maximum in urban areas decreases the likelihood of household identification tenfold. 
These errors are randomly and blindly applied to each original GPS point.  The new list of coordinates can be thought 
of from the perspective of each point having a circular error buffer zone (of 10km, 5km, or 2km) within which the raw 
value resides.

The geographic displacement methodology has been revised to the following set of steps which are conducted using a
Python script.  The Python script allows for a polygon layer to be specified as a displacement restrictor:

1)  Convert the coordinates from decimal degrees to meters using a fixed conversion factor from degrees to radians and
a scalor to correct for differences in the number of meters in a degree of latitude across the earth.

2)  Generate a random direction by generating angle between 0 and 360, and converting the angle from degrees to radians.

3)  Generate a random distance in meters of 0-2,000 meters for Urban points, and 0-5,000 meters for Rural points with 1% of 
rural points being given 0-10,000 meter distance.

4)  Generate the offset by applying trigonometry formulas (law of cosines) using the distance as the hypotenuse and the 
radians calculated in step 2.
	xOffset = math.sin(angle_radian) * distance
        yOffset = math.cos(angle_radian) * distance

5)  Add the offset to the original coordinate (in meters) to return the displaced coordinates.

6)  Re-convert the coordinates from meters to decimal degrees using a fixed conversion factor from radians to degrees and
a scalor to correct for differences in the number of meters in a degree of latitude across the earth.

7)  Determines whether the displaced coordinates are within the same polygon feature as the undisplaced coordinates.  Repeats
steps 1-6 as many times as necessary to generate displaced coordinates within the same polygon feature as the undisplaced
coordinates. 


If you have further questions please contact us at gpsrequests@dhsprogram.com.

