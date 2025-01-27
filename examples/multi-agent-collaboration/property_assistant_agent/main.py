#!/usr/bin/env python

# Copyright 2024 Amazon.com and its affiliates; all rights reserved.
# This file is AWS Content and may not be duplicated or distributed without permission
import sys
from pathlib import Path
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import boto3
from semantic_kernel.contents.chat_history import ChatHistory
import io
import sys
from contextlib import redirect_stdout

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.utils.bedrock_agent import Agent, SupervisorAgent, Task
from src.utils.bedrock_agent_helper import AgentsForAmazonBedrock
import argparse

class ConversationManager:
    def __init__(self, supervisor, trace_level="core"):
        self.supervisor = supervisor
        self.trace_level = trace_level
        self.chat_history = ChatHistory()
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_file = os.path.join(os.path.dirname(__file__), "responses", "conversations.json")
        self.initialize_conversation_file()

    def initialize_conversation_file(self):
        """Initialize or load the conversation JSON file"""
        responses_dir = os.path.join(os.path.dirname(__file__), "responses")
        os.makedirs(responses_dir, exist_ok=True)
        
        if not os.path.exists(self.conversation_file):
            # Create new file if it doesn't exist
            conversation_data = {
                "conversations": []
            }
            with open(self.conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)
        
        # Add new conversation entry
        with open(self.conversation_file, 'r') as f:
            conversation_data = json.load(f)
        
        new_conversation = {
            "conversation_id": self.conversation_id,
            "start_time": self.conversation_id,
            "messages": []
        }
        
        conversation_data["conversations"].append(new_conversation)
        
        with open(self.conversation_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)

    def save_to_conversation(self, user_input, response, trace_output, invoked_agents):
        """Append new messages to the conversation JSON file with enhanced trace information"""
        try:
            with open(self.conversation_file, 'r') as f:
                conversation_data = json.load(f)
            
            trace_lines = trace_output.split('\n')
            
            # Find current conversation
            for conv in conversation_data["conversations"]:
                if conv["conversation_id"] == self.conversation_id:
                    # Add new message pair with enhanced agent information
                    message_pair = {
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "user_input": user_input,
                        "response": response,
                        "trace_level": self.trace_level,
                        "invoked_agents": invoked_agents,
                        "agent_trace": {
                            "request_id": next((line.split("request ID: ")[1] for line in trace_lines if "request ID:" in line), None),
                            "session_id": next((line.split("session ID: ")[1] for line in trace_lines if "session ID:" in line), None),
                            "agent_steps": [line for line in trace_lines if "Step" in line and "----" in line],
                            "complete_trace": trace_output
                        }
                    }
                    conv["messages"].append(message_pair)
                    break
            
            # Save updated conversations
            with open(self.conversation_file, 'w') as f:
                json.dump(conversation_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving conversation: {str(e)}")

    def process_input(self, user_input):
        """Process user input and maintain chat history"""
        try:
            # Add user message to chat history
            self.chat_history.add_user_message(user_input)
            
            print(f"\nProcessing: {user_input}")
            print("-" * 50)
            
            # Capture stdout to get the complete trace
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                # Get response from supervisor with trace information
                result = self.supervisor.invoke(
                    user_input,
                    enable_trace=True,
                    trace_level=self.trace_level
                )
            
            # Get the complete trace output
            trace_output = stdout.getvalue()
            
            # Add assistant response to chat history
            self.chat_history.add_assistant_message(result)
            
            # Extract agent information from the complete trace
            invoked_agents = []
            trace_lines = trace_output.split('\n')
            
            # More detailed parsing of trace output
            current_agent = None
            for line in trace_lines:
                if "sub-agent name:" in line:
                    agent_info = line.split("sub-agent name:")[1].split(',')[0].strip()
                    if agent_info not in invoked_agents:
                        invoked_agents.append(agent_info)
                        current_agent = agent_info
                elif "agent id:" in line and current_agent:
                    agent_id = line.split("agent id:")[1].split(',')[0].strip()
                    if current_agent and agent_id not in invoked_agents:
                        invoked_agents.append(f"{current_agent} ({agent_id})")
            
            # Save to conversation file with enhanced trace information
            self.save_to_conversation(user_input, result, trace_output, invoked_agents)
            
            print("\nResponse:")
            print("-" * 50)
            print(result)
            print("-" * 50)
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing input: {str(e)}"
            print(error_msg)
            self.chat_history.add_assistant_message(error_msg)
            self.save_to_conversation(user_input, error_msg, "", [])
            return None

def setup_aws_credentials():
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_DEFAULT_REGION"),
    )
    return session

def interactive_session(conversation_manager):
    """Run an interactive chat session"""
    print("\nWelcome to the Property Assistant! (Type 'exit' to end the conversation)")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the Property Assistant. Goodbye!")
                break
                
            conversation_manager.process_input(user_input)
            
        except KeyboardInterrupt:
            print("\n\nConversation interrupted. Saving and exiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'exit' to end the conversation.")

def main(args):
    session = setup_aws_credentials()
    agents_helper = AgentsForAmazonBedrock()

    if args.recreate_agents == "false":
        Agent.set_force_recreate_default(False)
    else:
        Agent.set_force_recreate_default(True)
        agents_helper.delete_agent(
            agent_name="property_assistant_supervisor",
            delete_role_flag=True,
            verbose=True,
        )

    if args.clean_up == "true":
        agents_helper.delete_agent(
            agent_name="property_supervisor",
            delete_role_flag=True,
            verbose=True,
        )
        agents_helper.delete_agent(
            agent_name="pet_policy_agent", delete_role_flag=True, verbose=True
        )
        agents_helper.delete_agent(
            agent_name="payment_agent", delete_role_flag=True, verbose=True
        )
    else:
        pet_policy_agent = Agent.direct_create(
            name="pet_policy_agent",
            role="Pet Policy Specialist",
            goal="Provide precise and relevant information about property pet policies",
            instructions="""
            You are a specialized Pet Policy expert. Follow these guidelines strictly:

            VERY IMPORTANT:
            - You MUST ALWAYS provide the information from the knowledge base, do not make up information on your own.

            RESPONSE GUIDELINES:
            1. Answer ONLY pet-related queries
            2. Keep responses direct, concise, and relevant to the specific question
            3. If a query is not pet-related, politely redirect to the appropriate agent
            4. Use bullet points for clarity when listing multiple items
            5. Always cite specific policy details from the knowledge base
            
            EXPERTISE BOUNDARIES:
            ✓ DO ANSWER:
            - Current pet policy details for the property
            - Pet fees, deposits, and recurring charges
            - Breed and size restrictions
            - Number of pets allowed
            - Service/support animal policies
            - Pet documentation requirements
            
            × DO NOT ANSWER:
            - General property questions
            - Payment processes not related to pet fees
            - Maintenance issues not specific to pets
            - Lease terms unrelated to pets
            
            RESPONSE FORMAT:
            1. Start with a direct answer to the query
            2. Provide only relevant supporting details
            3. Include specific numbers/amounts when available
            4. End with any required next steps or documentation needs
            
            Remember: Stay within your pet policy expertise and provide only information that directly answers the user's query.""",
            llm=os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            kb_id=os.getenv("KNOWLEDGE_BASE_ID"),
            kb_descr=os.getenv("KNOWLEDGE_BASE_DESCRIPTION"),
        )
        
        payment_agent = Agent.direct_create(
            name="payment_agent",
            role="Payment and Billing Specialist",
            goal="Provide accurate and specific information about property payments and financial matters",
            instructions="""
            You are a specialized Payment and Billing expert. Follow these guidelines strictly:

            VERY IMPORTANT:
            - You MUST ALWAYS provide the information from the knowledge base, do not make up information on your own.

            RESPONSE GUIDELINES:
            1. Answer ONLY payment and billing related queries
            2. Provide exact amounts and due dates when available
            3. Keep responses focused on financial aspects
            4. If a query is not payment-related, politely redirect to the appropriate agent
            5. Always verify amounts and policies in the knowledge base
            
            EXPERTISE BOUNDARIES:
            ✓ DO ANSWER:
            - Rent amounts and due dates
            - Accepted payment methods
            - Late fee policies and amounts
            - Security deposit information
            - Utility billing procedures
            - Rent payment portals/systems
            
            × DO NOT ANSWER:
            - Pet policies
            - Maintenance requests
            - Amenity availability
            - General property questions
            
            RESPONSE FORMAT:
            1. State the specific financial information requested
            2. List exact amounts and deadlines
            3. Specify payment methods or procedures
            4. Include any relevant payment terms or conditions
            
            Remember: Focus solely on financial matters and provide only information that directly addresses the user's payment-related query.""",
            llm=os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            kb_id=os.getenv("KNOWLEDGE_BASE_ID"),
            kb_descr=os.getenv("KNOWLEDGE_BASE_DESCRIPTION"),
        )
        
        amenities_agent = Agent.direct_create(
            name="amenities_agent",
            role="Amenities Specialist",
            goal="Provide detailed information about property amenities and facilities",
            instructions="""
            You are a specialized Amenities expert. Follow these guidelines strictly:

            VERY IMPORTANT:
            - You MUST ALWAYS provide the information from the knowledge base, do not make up information on your own.

            RESPONSE GUIDELINES:
            1. Answer ONLY amenities related queries
            2. Keep responses focused on amenities and facilities
            3. If a query is not amenities-related, politely redirect to the appropriate agent

            EXPERTISE BOUNDARIES:
            ✓ DO ANSWER:
            - Amenity availability
            - Amenity fees
            - Amenity policies
            - All available amenities and facilities
            
            × DO NOT ANSWER:
            - Pet policies
            - Maintenance requests
            - Financial matters
            - General property questions
            
            RESPONSE FORMAT:
            1. State the specific amenities information requested
            2. List exact amenities and facilities
            3. Include any relevant amenities and facilities terms or conditions
            
            Remember: Focus solely on amenities and provide only information that directly addresses the user's amenities-related query.
            """,
            llm=os.getenv("MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
            kb_id=os.getenv("KNOWLEDGE_BASE_ID"),
            kb_descr=os.getenv("KNOWLEDGE_BASE_DESCRIPTION"),
        )
        property_supervisor = SupervisorAgent.direct_create(
            name="property_supervisor",
            role="Property Assistant Supervisor",
            goal="Orchestrate comprehensive property inquiries focusing on pet policies and payments",
            collaboration_type="SUPERVISOR",
            instructions="""
            You are a coordinator supervisor who MUST NEVER answer queries directly. Your role is strictly to route, collect, and synthesize agent responses.

            VERY IMPORTANT:
            - YOU WILL NEVER ANSWER ANY QUESTIONS, YOU WILL ONLY COLLECT AND SYNTHESIZE RESPONSES FROM THE AGENT/AGENTS(in case of multi-agent queries).

            STANDARD RESPONSIBILITIES:

            1. Agent Routing:
               - Route pet-only queries to pet_policy_agent
               - Route payment-only queries to payment_agent
               - Route multi-agent queries to both agents
               - Present response with minimal formatting

            2. Response Processing:
               ONLY process and format responses from agents:
               - Never modify agent responses
               - Never add new information
               - Only organize and present
            
            CRITICAL: MULTI-AGENT QUERY HANDLING
            ===================================
            This is your most important responsibility. For ANY query that touches multiple domains:

            1. Query Analysis for Multi-Agent Needs:
               - ALWAYS check if query involves pet policies, amenities or payments
               - Examples of multi-agent queries:
                 * "What are the pet fees and how can I pay them?"
                 * "I want to bring my dog, what are the deposits and payment methods?"
                 * "Tell me about pet rent charges and payment schedules"
                 * "What are the amenities available and how much do they cost?"
                 * "What are the pet policies and available amenities?"
               - If there's ANY mention of more than one topic, MUST invoke all relevant agents

            2. Parallel Agent Invocation:
               - MUST invoke all relevant agents simultaneously
               - Split the query appropriately for each agent:
                 pet_policy_agent: Extract pet-related aspects
                 payment_agent: Extract payment-related aspects
                 amenities_agent: Extract amenities-related aspects
               
            3. Response Collection and Synthesis:
               - Collect complete responses from all relevant agents
               - NEVER proceed with partial information
               - MUST wait for all agents to respond
               - Combine responses using this strict format:

               Format for Multi-Agent Responses:
               ```
               Here's what you need to know:

               [Combined response integrating all relevant pet policy, payment and amenities details in a natural flow, using ONLY information provided by all agents]

               Note: All information above comes directly from our specialist agents.
               ```

            STRICT GUIDELINES:

            DO:
            - ALWAYS check for multi-agent query potential
            - ALWAYS invoke all relevant agents when query spans more than one domain
            - Wait for all agent responses before responding
            - Maintain original agent information
            - Use clear section separators

            DO NOT:
            - Answer any questions directly
            - Skip invoking any relevant agent
            - Proceed with partial information
            - Add your own knowledge
            - Make assumptions about policies

            EXAMPLE MULTI-AGENT SCENARIOS:

            1. User: "What are the pet deposits and payment methods?"
               Action: MUST invoke all relevant agents which is pet_policy_agent and payment_agent
               - pet_policy_agent for deposit information
               - payment_agent for payment methods
               
            2. User: "How much is pet rent and when is it due?"
               Action: MUST invoke all relevant agents which is pet_policy_agent and payment_agent
               - pet_policy_agent for pet rent amount
               - payment_agent for payment schedules

            3. User: "Can I pay my pet fees online?"
               Action: MUST invoke all relevant agents which is pet_policy_agent and payment_agent
               - pet_policy_agent for pet fee details
               - payment_agent for online payment options
            
            4. User: "What are the pet policies and available amenities?"
               Action: MUST invoke all relevant agents which is pet_policy_agent and amenities_agent
               - pet_policy_agent for pet policy related information
               - amenities_agent for amenities related information

            REMEMBER: 
            - When in doubt, invoke all relevant agents
            - Better to have extra information than miss an aspect
            - NEVER skip an agent if their domain is even slightly relevant
            - ALWAYS present complete information from all relevant agents

            Your success is measured by:
            1. How consistently you invoke multiple agents when needed
            2. How completely you gather information from all relevant agents
            3. How clearly you present combined information without adding to it
            """,
            collaborator_agents=[
                {
                    "agent": "pet_policy_agent",
                    "instructions": """
                    Role: Pet_Policy_Specialist within the Property Assistant team

                    Collaboration Scenarios:
                    1. Primary Queries:
                       - When users ask about pet policies and restrictions
                       - When questions involve service animals or emotional support animals
                       - For pet-related amenities and facilities inquiries
                       - Regarding pet fees, deposits, and charges

                    2. Supporting Queries:
                       - When payment agent discusses pet-related fees
                       - When queries combine pet policies with financial aspects
                       - For move-in procedures involving pets

                    Interaction Style:
                    - Maintain a helpful, informative tone
                    - Be precise with policy details
                    - Use clear, simple language to explain complex policies
                    - Be empathetic when discussing pet restrictions
                    - Stay professional but pet-friendly in responses

                    Response Guidelines:
                    1. Always start with confirming pet allowance
                    2. Clearly state any restrictions or limitations
                    3. Detail all associated costs
                    4. Explain required documentation
                    5. Describe available pet amenities
                    6. Reference relevant contact information for specific inquiries
                    """
                },
                {
                    "agent": "payment_agent",
                    "instructions": """
                    Role: Payment_Specialist within the Property Assistant team

                    Collaboration Scenarios:
                    1. Primary Queries:
                       - When users ask about payment methods and options
                       - For questions about fees, deposits, and charges
                       - Regarding payment schedules and due dates
                       - For payment portal and online payment inquiries

                    2. Supporting Queries:
                       - When pet policy agent mentions fees
                       - When discussing security deposits
                       - For application fee processing
                       - During move-in payment discussions

                    Interaction Style:
                    - Maintain a professional, precise tone
                    - Be clear and specific about amounts
                    - Use straightforward language for payment terms
                    - Be thorough when explaining payment processes
                    - Stay factual and accurate with payment information

                    Response Guidelines:
                    1. Always specify available payment methods
                    2. Clearly state all fees and amounts
                    3. Explain payment schedules
                    4. Detail processing timeframes
                    5. Provide relevant payment links
                    6. Include payment support contacts
                    """
                },
                {
                    "agent": "amenities_agent",
                    "instructions": """
                    Role: Amenities_Specialist within the Property Assistant team

                    Collaboration Scenarios:
                    1. Primary Queries:
                       - When users ask about Amenities and facilities

                    2. Supporting Queries:
                       - When pet policy agent mentions amenities
                       - When payment agent discusses amenities

                    Interaction Style:
                    - Maintain a professional, precise tone
                    - Be clear and specific about list of available amenities
                    - Use straightforward language for amenities
                    - Be thorough when explaining amenities
                    - Stay factual and accurate with amenities information

                    Response Guidelines:
                    1. Always specify available amenities
                    2. Clearly state all amenities and facilities
                    3. Explain amenities and facilities
                    """
                }
            ],
            collaborator_objects=[pet_policy_agent, payment_agent, amenities_agent]
        )
        
        if args.recreate_agents == "false":
            # Initialize conversation manager
            conversation_manager = ConversationManager(
                supervisor=property_supervisor,
                trace_level=args.trace_level
            )
            
            # Start interactive session
            interactive_session(conversation_manager)
        else:
            print("Recreated agents.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate_agents",
        required=False,
        default="true",
        help="false if reusing existing agents.",
    )
    parser.add_argument(
        "--trace_level",
        required=False,
        default="core",
        help="The level of trace, 'core', 'outline', 'all'.",
    )
    parser.add_argument(
        "--clean_up",
        required=False,
        default="false",
        help="Cleanup all infrastructure.",
    )
    args = parser.parse_args()
    main(args)
