# compute_isochrones.R — r5r-based 15-minute walking isochrone computation
#
# Called from Python via subprocess.
# Usage: Rscript compute_isochrones.R <osm_pbf> <points_csv> <output_gpkg> [threshold_min] [walk_speed]
#
# Input:  CSV with columns: id, lat, lon (home grid cell centroids)
# Output: GeoPackage with columns: id, geometry (isochrone polygons)
