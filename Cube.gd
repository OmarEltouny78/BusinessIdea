extends KinematicBody


# Declare member variables here. Examples:
# var a = 2
# var b = "text"

var m_animPlayer
var m_godotLogoAnim
# Called when the node enters the scene tree for the first time.
func _ready():
	m_animPlayer = AnimationPlayer.new()
	add_child( m_animPlayer )
	m_godotLogoAnim = Animation.new()
	m_animPlayer.add_animation( "godotAnim", m_godotLogoAnim )
	m_animPlayer.set_current_animation( "godotAnim" )
	m_godotLogoAnim.add_track( 0 )
	m_godotLogoAnim.track_set_path( 0, "/root/Spatial/Cube:rotation:x" )
	m_godotLogoAnim.track_insert_key( 0, 0.0, 0.0 )
	m_godotLogoAnim.track_insert_key( 0.0, 0.2, 10.0 )
	m_animPlayer.play( "godotAnim" )
	print(translation.x)


# Called every frame. 'delta' is the elapsed time since the previous frame.
#func _process(delta):
#	pass
