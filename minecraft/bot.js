/**
 * NPCPU Mineflayer Bot
 *
 * A Minecraft bot that communicates with NPCPU Python code via JSON stdio.
 * Uses Mineflayer for world interaction and pathfinder for navigation.
 *
 * Setup:
 *   npm install mineflayer mineflayer-pathfinder
 *
 * The bot reads JSON commands from stdin and writes responses to stdout.
 */

const mineflayer = require('mineflayer');
const pathfinder = require('mineflayer-pathfinder').pathfinder;
const Movements = require('mineflayer-pathfinder').Movements;
const { GoalNear, GoalBlock, GoalXZ } = require('mineflayer-pathfinder').goals;
const readline = require('readline');

// Bot instance
let bot = null;
let mcData = null;
let defaultMove = null;

// Command handlers
const handlers = {};

// Initialize readline for JSON communication
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    terminal: false
});

// Send JSON message to Python
function send(type, data) {
    const message = JSON.stringify({ type, ...data });
    console.log(message);
}

// Send response to a command
function respond(commandId, success, data = null, error = null) {
    send('response', {
        command_id: commandId,
        success,
        data,
        error
    });
}

// Send state update
function sendStateUpdate() {
    if (!bot || !bot.entity) return;

    send('state_update', {
        state: {
            position: bot.entity.position,
            health: bot.health,
            food: bot.food,
            saturation: bot.foodSaturation,
            experience: bot.experience.level,
            gamemode: bot.game.gameMode,
            is_alive: bot.health > 0,
            on_ground: bot.entity.onGround,
            inventory: bot.inventory.items().map(item => ({
                name: item.name,
                count: item.count,
                slot: item.slot
            }))
        }
    });
}

// Send world update
function sendWorldUpdate() {
    if (!bot) return;

    const nearbyEntities = Object.values(bot.entities)
        .filter(e => e !== bot.entity && e.position)
        .map(e => ({
            id: e.id,
            type: e.name || e.type,
            name: e.username || e.displayName || null,
            position: e.position,
            health: e.health || null,
            is_hostile: isHostile(e),
            is_player: e.type === 'player',
            distance: bot.entity.position.distanceTo(e.position)
        }))
        .filter(e => e.distance < 32)
        .sort((a, b) => a.distance - b.distance);

    send('world_update', {
        world: {
            nearby_entities: nearbyEntities,
            time: bot.time.timeOfDay,
            weather: bot.isRaining ? (bot.thunderState > 0 ? 'thunder' : 'rain') : 'clear',
            biome: bot.world.getBiome ? bot.world.getBiome(bot.entity.position) : 'unknown',
            dimension: bot.game.dimension
        }
    });
}

// Check if entity is hostile
function isHostile(entity) {
    const hostiles = [
        'zombie', 'skeleton', 'creeper', 'spider', 'enderman', 'witch',
        'slime', 'phantom', 'drowned', 'husk', 'stray', 'pillager',
        'vindicator', 'evoker', 'ravager', 'blaze', 'ghast', 'wither_skeleton'
    ];
    return hostiles.includes(entity.name || entity.type);
}

// Command: Connect to server
handlers.connect = async (params, commandId) => {
    try {
        const { host, port, username, version, auth } = params;

        bot = mineflayer.createBot({
            host: host || 'localhost',
            port: port || 25565,
            username: username || 'NPCPU_Bot',
            version: version || false,
            auth: auth || 'offline'
        });

        // Load plugins
        bot.loadPlugin(pathfinder);

        // Setup event handlers
        bot.once('spawn', () => {
            mcData = require('minecraft-data')(bot.version);
            defaultMove = new Movements(bot, mcData);
            bot.pathfinder.setMovements(defaultMove);

            respond(commandId, true, { message: 'Connected and spawned' });
            send('event', { event: 'spawn', data: {} });

            // Start periodic updates
            setInterval(sendStateUpdate, 1000);
            setInterval(sendWorldUpdate, 2000);
        });

        bot.on('health', () => {
            sendStateUpdate();
            if (bot.health <= 0) {
                send('event', { event: 'death', data: {} });
            }
        });

        bot.on('chat', (username, message) => {
            send('event', { event: 'chat', data: { username, message } });
        });

        bot.on('entityHurt', (entity) => {
            if (entity === bot.entity) {
                send('event', { event: 'damage', data: { health: bot.health } });
            }
        });

        bot.on('entitySpawn', (entity) => {
            if (isHostile(entity) && entity.position.distanceTo(bot.entity.position) < 16) {
                send('event', { event: 'entity_spawn', data: {
                    id: entity.id,
                    type: entity.name,
                    position: entity.position,
                    is_hostile: true
                }});
            }
        });

        bot.on('error', (err) => {
            send('event', { event: 'error', data: { message: err.message } });
        });

        bot.on('kicked', (reason) => {
            send('event', { event: 'kicked', data: { reason } });
        });

    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Disconnect
handlers.disconnect = async (params, commandId) => {
    if (bot) {
        bot.quit();
        bot = null;
    }
    respond(commandId, true);
};

// Command: Move to position
handlers.move_to = async (params, commandId) => {
    try {
        const { x, y, z } = params;
        const goal = new GoalNear(x, y, z, 1);

        await bot.pathfinder.goto(goal);
        respond(commandId, true, { position: bot.entity.position });
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Look at position
handlers.look_at = async (params, commandId) => {
    try {
        const { x, y, z } = params;
        await bot.lookAt({ x, y, z });
        respond(commandId, true);
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Attack entity
handlers.attack = async (params, commandId) => {
    try {
        const { entity_id } = params;
        const entity = bot.entities[entity_id];

        if (!entity) {
            respond(commandId, false, null, 'Entity not found');
            return;
        }

        await bot.attack(entity);
        respond(commandId, true);
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Attack nearest hostile
handlers.attack_nearest_hostile = async (params, commandId) => {
    try {
        const hostile = Object.values(bot.entities)
            .filter(e => isHostile(e) && e.position)
            .sort((a, b) =>
                bot.entity.position.distanceTo(a.position) -
                bot.entity.position.distanceTo(b.position)
            )[0];

        if (!hostile) {
            respond(commandId, false, null, 'No hostile entities nearby');
            return;
        }

        // Move close and attack
        const goal = new GoalNear(hostile.position.x, hostile.position.y, hostile.position.z, 2);
        await bot.pathfinder.goto(goal);
        await bot.attack(hostile);

        respond(commandId, true, { target: hostile.name });
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Dig block
handlers.dig = async (params, commandId) => {
    try {
        const { x, y, z } = params;
        const block = bot.blockAt({ x, y, z });

        if (!block || block.name === 'air') {
            respond(commandId, false, null, 'No block at position');
            return;
        }

        await bot.dig(block);
        respond(commandId, true, { block: block.name });
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Place block
handlers.place = async (params, commandId) => {
    try {
        const { position, block: blockName } = params;

        // Find the block in inventory
        const item = bot.inventory.items().find(i => i.name.includes(blockName));
        if (!item) {
            respond(commandId, false, null, `No ${blockName} in inventory`);
            return;
        }

        // Equip and place
        await bot.equip(item, 'hand');
        const referenceBlock = bot.blockAt({ x: position.x, y: position.y - 1, z: position.z });
        await bot.placeBlock(referenceBlock, { x: 0, y: 1, z: 0 });

        respond(commandId, true);
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Eat food
handlers.eat = async (params, commandId) => {
    try {
        const foods = ['cooked_beef', 'cooked_porkchop', 'bread', 'cooked_chicken', 'apple', 'golden_apple'];
        const food = bot.inventory.items().find(i => foods.some(f => i.name.includes(f)));

        if (!food) {
            respond(commandId, false, null, 'No food in inventory');
            return;
        }

        await bot.equip(food, 'hand');
        await bot.consume();

        respond(commandId, true, { item: food.name });
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Craft item
handlers.craft = async (params, commandId) => {
    try {
        const { recipe: recipeName, count } = params;

        const recipe = bot.recipesFor(mcData.itemsByName[recipeName]?.id)[0];
        if (!recipe) {
            respond(commandId, false, null, 'Recipe not found');
            return;
        }

        await bot.craft(recipe, count || 1);
        respond(commandId, true);
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Send chat message
handlers.chat = async (params, commandId) => {
    try {
        const { message } = params;
        bot.chat(message);
        respond(commandId, true);
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Get current state
handlers.get_state = async (params, commandId) => {
    try {
        respond(commandId, true, {
            position: bot.entity.position,
            health: bot.health,
            food: bot.food,
            saturation: bot.foodSaturation,
            experience: bot.experience.level,
            gamemode: bot.game.gameMode,
            is_alive: bot.health > 0,
            inventory: bot.inventory.items().map(item => ({
                name: item.name,
                count: item.count,
                slot: item.slot
            }))
        });
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Get world state
handlers.get_world_state = async (params, commandId) => {
    try {
        const nearbyEntities = Object.values(bot.entities)
            .filter(e => e !== bot.entity && e.position)
            .map(e => ({
                id: e.id,
                type: e.name || e.type,
                name: e.username || null,
                position: e.position,
                is_hostile: isHostile(e),
                is_player: e.type === 'player',
                distance: bot.entity.position.distanceTo(e.position)
            }))
            .filter(e => e.distance < 32);

        respond(commandId, true, {
            nearby_entities: nearbyEntities,
            time: bot.time.timeOfDay,
            weather: bot.isRaining ? 'rain' : 'clear',
            dimension: bot.game.dimension
        });
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Find blocks of type
handlers.find_blocks = async (params, commandId) => {
    try {
        const { block: blockName, radius } = params;
        const blockType = mcData.blocksByName[blockName];

        if (!blockType) {
            respond(commandId, false, null, `Unknown block: ${blockName}`);
            return;
        }

        const blocks = bot.findBlocks({
            matching: blockType.id,
            maxDistance: radius || 32,
            count: 20
        });

        const positions = blocks.map(pos => ({
            x: pos.x,
            y: pos.y,
            z: pos.z,
            distance: bot.entity.position.distanceTo(pos)
        }));

        respond(commandId, true, positions);
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Command: Find entities
handlers.find_entities = async (params, commandId) => {
    try {
        const { type, hostile_only } = params;

        let entities = Object.values(bot.entities)
            .filter(e => e !== bot.entity && e.position);

        if (type) {
            entities = entities.filter(e => e.name === type || e.type === type);
        }

        if (hostile_only) {
            entities = entities.filter(e => isHostile(e));
        }

        const result = entities.map(e => ({
            id: e.id,
            type: e.name || e.type,
            name: e.username || null,
            position: e.position,
            is_hostile: isHostile(e),
            distance: bot.entity.position.distanceTo(e.position)
        }));

        respond(commandId, true, result);
    } catch (error) {
        respond(commandId, false, null, error.message);
    }
};

// Process incoming commands
rl.on('line', async (line) => {
    try {
        const command = JSON.parse(line);
        const { id, action, params } = command;

        if (handlers[action]) {
            await handlers[action](params, id);
        } else {
            respond(id, false, null, `Unknown action: ${action}`);
        }
    } catch (error) {
        console.error('Command error:', error.message);
    }
});

// Handle process termination
process.on('SIGINT', () => {
    if (bot) bot.quit();
    process.exit();
});

process.on('SIGTERM', () => {
    if (bot) bot.quit();
    process.exit();
});

// Startup message
console.error('NPCPU Mineflayer Bot started. Waiting for commands...');
